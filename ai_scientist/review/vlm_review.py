"""VLM-based paper review functionality for figures and images."""

import os
import hashlib
import pymupdf
import re
import base64

from ai_scientist.vlm import (
    get_response_from_vlm,
    extract_json_between_markers,
)
from ai_scientist.utils.model_params import build_token_params
from ai_scientist.review.pdf_utils import load_paper
from ai_scientist.prompt_loader import load_prompt

# Load prompt templates
reviewer_system_prompt_base = load_prompt("review/llm/system_base")
IMG_CAP_REF_REVIEW_PROMPT_TEMPLATE = load_prompt(
    "review/vlm/img_cap_ref_review"
)
IMG_CAP_SELECTION_PROMPT_TEMPLATE = load_prompt(
    "review/vlm/img_cap_selection"
)
IMG_REVIEW_PROMPT_TEMPLATE = load_prompt("review/vlm/img_review")
DUPLICATE_FIGURES_SYSTEM_PROMPT = load_prompt(
    "review/vlm/duplicate_figures_system"
).strip()
DUPLICATE_FIGURES_USER_PROMPT = load_prompt(
    "review/vlm/duplicate_figures_user"
).strip()


def encode_image_to_base64(image_data):
    """Encode image data to base64 string.

    Args:
        image_data: Image data as string path, list of bytes, or bytes.

    Returns:
        Base64 encoded string.

    Raises:
        TypeError: If image_data type is not supported.
    """
    if isinstance(image_data, str):
        with open(image_data, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image_data, list):
        return base64.b64encode(image_data[0]).decode("utf-8")
    elif isinstance(image_data, bytes):
        return base64.b64encode(image_data).decode("utf-8")
    else:
        raise TypeError(f"Unsupported image data type: {type(image_data)}")


def extract_figure_screenshots(
    pdf_path,
    img_folder_path,
    num_pages=None,
    min_text_length=50,
    min_vertical_gap=30,
):
    """Extract screenshots for figure captions from a PDF.

    Looks for figure captions ("Figure X." or "Figure X:") and extracts
    the corresponding figure images. Also gathers text blocks mentioning
    each figure.

    Args:
        pdf_path: Path to the PDF file.
        img_folder_path: Directory to save extracted images.
        num_pages: Optional limit on pages to process.
        min_text_length: Minimum text length for reference blocks.
        min_vertical_gap: Minimum vertical gap between caption and figure.

    Returns:
        List of dictionaries containing figure info (img_name, caption, images, main_text_figrefs).
    """
    os.makedirs(img_folder_path, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    page_range = (
        range(len(doc)) if num_pages is None else range(min(num_pages, len(doc)))
    )

    # Extract all text blocks from the document
    text_blocks = []
    for page_num in page_range:
        page = doc[page_num]
        try:
            blocks = page.get_text("blocks")
            for b in blocks:
                txt = b[4].strip()
                if txt:
                    bbox = pymupdf.Rect(b[0], b[1], b[2], b[3])
                    text_blocks.append({"page": page_num, "bbox": bbox, "text": txt})
        except Exception as e:
            print(f"Error extracting text from page {page_num}: {e}")

    # Regex for figure captions
    figure_caption_pattern = re.compile(
        r"^(?:Figure)\s+(?P<fig_label>"
        r"(?:\d+"  # "1", "11", ...
        r"|[A-Za-z]+\.\d+"  # "A.1", "S2.3"
        r"|\(\s*[A-Za-z]+\s*\)\.\d+"  # "(A).2"
        r")"
        r")(?:\.|:)",  # Must end with "." or ":"
        re.IGNORECASE,
    )

    # Detect sub-figure captions (e.g. "(a)")
    subfigure_pattern = re.compile(r"\(\s*[a-zA-Z]\s*\)")

    def is_subfigure_caption(txt):
        return bool(subfigure_pattern.search(txt))

    result_pairs = []

    for page_num in page_range:
        page = doc[page_num]
        page_rect = page.rect

        # All text blocks for this page
        page_blocks = [b for b in text_blocks if b["page"] == page_num]
        # Sort top-to-bottom
        page_blocks.sort(key=lambda b: b["bbox"].y0)

        # Find figure captions
        for blk in page_blocks:
            caption_text = blk["text"]
            m = figure_caption_pattern.match(caption_text)
            if not m:
                continue

            fig_label = m.group("fig_label")
            fig_x0, fig_y0, fig_x1, fig_y1 = blk["bbox"]

            # Find a large text block above the caption (on the same page)
            above_blocks = []
            for ab in page_blocks:
                if ab["bbox"].y1 < fig_y0:
                    ab_height_gap = fig_y0 - ab["bbox"].y1
                    overlap_x = min(fig_x1, ab["bbox"].x1) - max(fig_x0, ab["bbox"].x0)
                    width_min = min((fig_x1 - fig_x0), (ab["bbox"].x1 - ab["bbox"].x0))
                    horiz_overlap_ratio = (
                        overlap_x / float(width_min) if width_min > 0 else 0.0
                    )

                    if (
                        len(ab["text"]) >= min_text_length
                        and not is_subfigure_caption(ab["text"])
                        and ab_height_gap >= min_vertical_gap
                        and horiz_overlap_ratio > 0.3
                    ):
                        above_blocks.append(ab)

            # Pick the block with the largest bottom edge
            if above_blocks:
                above_block = max(above_blocks, key=lambda b: b["bbox"].y1)
                clip_top = above_block["bbox"].y1
            else:
                clip_top = page_rect.y0

            clip_left = fig_x0
            clip_right = fig_x1
            clip_bottom = fig_y0

            # Create figure screenshot
            if (clip_bottom > clip_top) and (clip_right > clip_left):
                clip_rect = pymupdf.Rect(clip_left, clip_top, clip_right, clip_bottom)
                pix = page.get_pixmap(clip=clip_rect, dpi=150)

                fig_label_escaped = re.escape(fig_label)
                fig_hash = hashlib.md5(
                    f"figure_{fig_label_escaped}_{page_num}_{clip_rect}".encode()
                ).hexdigest()[:10]
                fig_filename = (
                    f"figure_{fig_label_escaped}_Page_{page_num+1}_{fig_hash}.png"
                )
                fig_filepath = os.path.join(img_folder_path, fig_filename)
                pix.save(fig_filepath)

                # Find references across the entire document
                fig_label_escaped = re.escape(fig_label)
                main_text_figure_pattern = re.compile(
                    rf"(?:Fig(?:\.|-\s*ure)?|Figure)\s*{fig_label_escaped}(?![0-9A-Za-z])",
                    re.IGNORECASE,
                )

                references_in_doc = []
                for tb in text_blocks:
                    if tb is blk:
                        continue
                    if main_text_figure_pattern.search(tb["text"]):
                        references_in_doc.append(tb["text"])

                result_pairs.append(
                    {
                        "img_name": f"figure_{fig_label_escaped}",
                        "caption": caption_text,
                        "images": [fig_filepath],
                        "main_text_figrefs": references_in_doc,
                    }
                )

    return result_pairs


def extract_abstract(text):
    """Extract the abstract section from paper text.

    Args:
        text: Full paper text with markdown headings.

    Returns:
        Abstract text content, or empty string if not found.
    """
    lines = text.split("\n")
    heading_pattern = re.compile(r"^\s*#+\s*(.*)$")

    abstract_start = None
    for i, line in enumerate(lines):
        match = heading_pattern.match(line)
        if match:
            heading_text = match.group(1)
            if "abstract" in heading_text.lower():
                abstract_start = i
                break

    if abstract_start is None:
        return ""

    abstract_lines = []
    for j in range(abstract_start + 1, len(lines)):
        if heading_pattern.match(lines[j]):
            break
        abstract_lines.append(lines[j])

    abstract_text = "\n".join(abstract_lines).strip()
    return abstract_text


def generate_vlm_img_cap_ref_review(img, abstract, model, client):
    """Generate a VLM review of a figure with caption and references.

    Args:
        img: Dictionary with 'images', 'caption', and 'main_text_figrefs'.
        abstract: Paper abstract text.
        model: VLM model to use.
        client: VLM client instance.

    Returns:
        Review dictionary from VLM.
    """
    prompt = IMG_CAP_REF_REVIEW_PROMPT_TEMPLATE.format(
        abstract=abstract,
        caption=img["caption"],
        main_text_figrefs=img["main_text_figrefs"],
    )
    content, _ = get_response_from_vlm(
        prompt, img["images"], client, model, reviewer_system_prompt_base
    )
    img_cap_ref_review_json = extract_json_between_markers(content)
    return img_cap_ref_review_json


def generate_vlm_img_review(img, model, client):
    """Generate a VLM review of just the image.

    Args:
        img: Dictionary with 'images' key.
        model: VLM model to use.
        client: VLM client instance.

    Returns:
        Review dictionary from VLM.
    """
    prompt = IMG_REVIEW_PROMPT_TEMPLATE
    content, _ = get_response_from_vlm(
        prompt, img["images"], client, model, reviewer_system_prompt_base
    )
    img_review_json = extract_json_between_markers(content)
    return img_review_json


def perform_imgs_cap_ref_review(client, client_model, pdf_path):
    """Perform VLM review of all figures in a PDF.

    Args:
        client: VLM client instance.
        client_model: VLM model name.
        pdf_path: Path to the PDF file.

    Returns:
        Dictionary mapping figure names to their reviews.
    """
    paper_txt = load_paper(pdf_path)
    img_folder_path = os.path.join(
        os.path.dirname(pdf_path),
        f"{os.path.splitext(os.path.basename(pdf_path))[0]}_imgs",
    )
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    img_pairs = extract_figure_screenshots(pdf_path, img_folder_path)
    img_reviews = {}
    abstract = extract_abstract(paper_txt)
    for img in img_pairs:
        review = generate_vlm_img_cap_ref_review(img, abstract, client_model, client)
        img_reviews[img["img_name"]] = review
    return img_reviews


def detect_duplicate_figures(client, client_model, pdf_path):
    """Detect duplicate figures in a PDF.

    Args:
        client: VLM client instance.
        client_model: VLM model name.
        pdf_path: Path to the PDF file.

    Returns:
        Analysis result from VLM, or error dict if failed.
    """
    paper_txt = load_paper(pdf_path)
    img_folder_path = os.path.join(
        os.path.dirname(pdf_path),
        f"{os.path.splitext(os.path.basename(pdf_path))[0]}_imgs",
    )
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    img_pairs = extract_figure_screenshots(pdf_path, img_folder_path)

    messages = [
        {
            "role": "system",
            "content": DUPLICATE_FIGURES_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": DUPLICATE_FIGURES_USER_PROMPT,
                }
            ],
        },
    ]

    # Add images in the correct format
    for img_info in img_pairs:
        messages[1]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_to_base64(img_info['images'][0])}"
                },
            }
        )

    try:
        response = client.chat.completions.create(
            model=client_model,
            messages=messages,
            **build_token_params(client_model, 1000),
        )

        analysis = response.choices[0].message.content
        return analysis

    except Exception as e:
        print(f"Error analyzing images: {e}")
        return {"error": str(e)}


def generate_vlm_img_selection_review(
    img, abstract, model, client, reflection_page_info
):
    """Generate a VLM selection review for a figure.

    Args:
        img: Dictionary with figure info.
        abstract: Paper abstract text.
        model: VLM model to use.
        client: VLM client instance.
        reflection_page_info: Additional page/reflection context.

    Returns:
        Review dictionary from VLM.
    """
    prompt = IMG_CAP_SELECTION_PROMPT_TEMPLATE.format(
        abstract=abstract,
        caption=img["caption"],
        main_text_figrefs=img["main_text_figrefs"],
        reflection_page_info=reflection_page_info,
    )
    content, _ = get_response_from_vlm(
        prompt, img["images"], client, model, reviewer_system_prompt_base
    )
    img_cap_ref_review_json = extract_json_between_markers(content)
    return img_cap_ref_review_json


def perform_imgs_cap_ref_review_selection(
    client, client_model, pdf_path, reflection_page_info
):
    """Perform VLM selection review of all figures in a PDF.

    Args:
        client: VLM client instance.
        client_model: VLM model name.
        pdf_path: Path to the PDF file.
        reflection_page_info: Additional page/reflection context.

    Returns:
        Dictionary mapping figure names to their reviews.
    """
    paper_txt = load_paper(pdf_path)
    img_folder_path = os.path.join(
        os.path.dirname(pdf_path),
        f"{os.path.splitext(os.path.basename(pdf_path))[0]}_imgs",
    )
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)
    img_pairs = extract_figure_screenshots(pdf_path, img_folder_path)
    img_reviews = {}
    abstract = extract_abstract(paper_txt)
    for img in img_pairs:
        review = generate_vlm_img_selection_review(
            img, abstract, client_model, client, reflection_page_info
        )
        img_reviews[img["img_name"]] = review
    return img_reviews
