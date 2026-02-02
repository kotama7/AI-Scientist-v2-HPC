const bgCol = "#FFFFFF";
const accentCol = "#1a439e";

hljs.initHighlightingOnLoad();

// Function to update background color globally
function updateBackgroundColor(color) {
  // Update the JS variable
  window.bgColCurrent = color;

  // Update body background
  document.body.style.backgroundColor = color;

  // Update canvas container background
  const canvasContainer = document.getElementById('canvas-container');
  if (canvasContainer) {
    canvasContainer.style.backgroundColor = color;
  }
}

// Store tree data for each stage
const stageData = {
  Stage_1: null,
  Stage_2: null,
  Stage_3: null,
  Stage_4: null
};

const defaultStageDirMap = {
  Stage_1: 'stage_1_initial_implementation_1_preliminary',
  Stage_2: 'stage_2_baseline_tuning_1_first_attempt',
  Stage_3: 'stage_3_creative_research_1_first_attempt',
  Stage_4: 'stage_4_ablation_studies_1_first_attempt'
};

// Keep track of current selected stage
let currentStage = null;
let currentSketch = null;
let availableStages = [];
let memoryPhaseIndex = 0;
let memoryPhaseKeys = [];
let memoryEventsByPhase = {};
let memoryCurrentFilter = 'all';

// Memory operation categorization mapping
const MEMORY_OP_CATEGORIES = {
  // Read operations
  'get_core': 'reads',
  'mem_core_get': 'reads',
  'render_for_prompt': 'reads',
  'mem_node_read': 'reads',
  'mem_archival_search': 'reads',
  'mem_archival_get': 'reads',
  'mem_recall_search': 'reads',
  'retrieve_archival': 'reads',
  // Write operations
  'set_core': 'writes',
  'mem_core_set': 'writes',
  'write_archival': 'writes',
  'mem_archival_write': 'writes',
  'mem_archival_update': 'writes',
  'mem_node_write': 'writes',
  'write_event': 'writes',
  // Delete operations
  'core_evict': 'deletes',
  'core_delete': 'deletes',
  'core_digest_compact': 'deletes',
  // Fork operations
  'mem_node_fork': 'forks',
  // Recall operations
  'mem_recall_append': 'recalls',
  // Maintenance operations
  'consolidate_recall_events': 'maintenance',
  'check_memory_pressure': 'maintenance',
  'auto_consolidate_memory': 'maintenance',
  'evaluate_importance_with_llm': 'maintenance',
};

// Category display configuration
const CATEGORY_CONFIG = {
  'reads': { icon: '📖', label: 'Reads', color: '#4dabf7' },
  'writes': { icon: '💾', label: 'Writes', color: '#69db7c' },
  'deletes': { icon: '🗑️', label: 'Deletes', color: '#ff6b6b' },
  'forks': { icon: '🌿', label: 'Forks', color: '#da77f2' },
  'recalls': { icon: '🔄', label: 'Recalls', color: '#ffd43b' },
  'maintenance': { icon: '🔧', label: 'Maintenance', color: '#adb5bd' },
  'other': { icon: '❓', label: 'Other', color: '#868e96' },
};

function inferStageIdFromPath(pathname) {
  const match = pathname.match(/stage_(\d+)/);
  return match ? `Stage_${match[1]}` : null;
}

function inferLogDirPath(pathname) {
  const parts = pathname.split('/');
  const stageIndex = parts.findIndex(part => part.startsWith('stage_'));
  if (stageIndex > 0) {
    const prefix = parts.slice(0, stageIndex).join('/');
    return prefix || '.';
  }
  const prefix = parts.slice(0, -1).join('/');
  return prefix || '.';
}

function addAvailableStage(stageId) {
  if (!availableStages.includes(stageId)) {
    availableStages.push(stageId);
  }
}

// Class definitions for nodes and edges
class Node {
  constructor(x, y, id, isRoot = false, isBestNode = false) {
    this.x = x;
    this.y = y;
    this.id = id;
    this.visible = isRoot; // Only root nodes are visible initially
    this.appearProgress = 0;
    this.popEffect = 0;
    this.selected = false;
    this.isRootNode = isRoot;
    this.isBestNode = isBestNode;
  }

  update() {
    if (this.visible) {
      // Handle the main appearance animation
      if (this.appearProgress < 1) {
        this.appearProgress += 0.06;

        // When we reach full size, trigger the pop effect
        if (this.appearProgress >= 1) {
          this.appearProgress = 1; // Cap at 1
          this.popEffect = 1; // Start the pop effect
        }
      }

      // Handle the pop effect animation
      if (this.popEffect > 0) {
        this.popEffect -= 0.15; // Control how quickly it shrinks back
        if (this.popEffect < 0) this.popEffect = 0; // Don't go negative
      }
    }
  }

  startAnimation() {
    this.visible = true;
  }

  color() {
    if (this.selected) {
      return accentCol; // Use the global accent color variable for selected node
    }
    if (this.isBestNode) {
      return '#f59f00'; // Gold color for best node
    }
    return '#4263eb'; // Default blue color
  }

  render(p5) {
    if (this.visible) {
      const popBonus = this.popEffect * 0.1;
      const nodeScale = p5.map(this.appearProgress, 0, 1, 0, 1) + popBonus;
      const alpha = p5.map(this.appearProgress, 0, 1, 0, 255);

      p5.push();
      p5.translate(this.x, this.y);

      // Shadow effect
      p5.noStroke();
      p5.rectMode(p5.CENTER);

      for (let i = 1; i <= 4; i++) {
        p5.fill(0, 0, 0, alpha * 0.06);
        p5.rect(i, i, 30 * nodeScale, 30 * nodeScale, 10);
      }

      // Main square - use node's color with alpha
      let nodeColor = p5.color(this.color());
      nodeColor.setAlpha(alpha);
      p5.fill(nodeColor);
      p5.rect(0, 0, 30 * nodeScale, 30 * nodeScale, 10);

      // Draw star icon if this is the best node
      if (this.isBestNode && this.appearProgress >= 1) {
        p5.fill(255);
        p5.noStroke();
        p5.beginShape();
        for (let a = 0; a < 5; a++) {
          let angle = p5.TWO_PI * a / 5 - p5.HALF_PI;
          p5.vertex(p5.cos(angle) * 8 * nodeScale, p5.sin(angle) * 8 * nodeScale);
          angle += p5.TWO_PI / 10;
          p5.vertex(p5.cos(angle) * 4 * nodeScale, p5.sin(angle) * 4 * nodeScale);
        }
        p5.endShape(p5.CLOSE);
      }
      // Draw checkmark icon if the node is selected (and not the best node)
      else if (this.selected && this.appearProgress >= 1) {
        p5.stroke(255);
        p5.strokeWeight(2 * nodeScale);
        p5.noFill();
        // Draw checkmark
        p5.beginShape();
        p5.vertex(-8, 0);
        p5.vertex(-3, 5);
        p5.vertex(8, -6);
        p5.endShape();
      }

      p5.pop();
    }
  }

  isMouseOver(p5) {
    return this.visible &&
           p5.mouseX > this.x - 15 &&
           p5.mouseX < this.x + 15 &&
           p5.mouseY > this.y - 15 &&
           p5.mouseY < this.y + 15;
  }

  // Connect this node to a child node
  child(childNode) {
    // Create an edge from this node to the child
    let isLeft = childNode.x < this.x;
    let isRight = childNode.x > this.x;
    let edge = new Edge(this, childNode, isLeft, isRight);
    return edge;
  }
}

class Edge {
  constructor(parent, child, isLeft, isRight) {
    this.parent = parent;
    this.child = child;
    this.isLeft = isLeft;
    this.isRight = isRight;
    this.progress = 0;

    // Calculate the midpoint where branching occurs
    this.midY = parent.y + (child.y - parent.y) * 0.6;

    // Use the actual child x-coordinate
    // This ensures the edge will connect directly to the child node
    this.branchX = child.x;
  }

  update() {
    if (this.parent.visible && this.progress < 1) {
      this.progress += 0.01; // Adjust animation speed
    }
    if (this.progress >= 1) {
      this.child.visible = true;
    }
  }

  color() {
    return this.child.color();
  }

  render(p5) {
    if (!this.parent.visible) return;

    // Calculate path lengths
    const verticalDist1 = this.midY - this.parent.y;
    const horizontalDist = Math.abs(this.branchX - this.parent.x);
    const verticalDist2 = this.child.y - this.midY;
    const totalLength = verticalDist1 + horizontalDist + verticalDist2;

    // Calculate how much of each segment to draw
    const currentLength = totalLength * this.progress;

    p5.stroke(180, 190, 205);
    p5.strokeWeight(1.5);
    p5.noFill();

    // Always draw the first vertical segment from parent
    if (currentLength > 0) {
      const firstSegmentLength = Math.min(currentLength, verticalDist1);
      const currentMidY = p5.lerp(this.parent.y, this.midY, firstSegmentLength / verticalDist1);
      p5.line(this.parent.x, this.parent.y, this.parent.x, currentMidY);
    }

    if (currentLength > verticalDist1) {
      // Draw second segment (horizontal)
      const secondSegmentLength = Math.min(currentLength - verticalDist1, horizontalDist);
      const currentBranchX = p5.lerp(this.parent.x, this.branchX, secondSegmentLength / horizontalDist);
      p5.line(this.parent.x, this.midY, currentBranchX, this.midY);

      if (currentLength > verticalDist1 + horizontalDist) {
        // Draw third segment (vertical to child)
        const thirdSegmentLength = currentLength - verticalDist1 - horizontalDist;
        const currentChildY = p5.lerp(this.midY, this.child.y, thirdSegmentLength / verticalDist2);
        p5.line(this.branchX, this.midY, this.branchX, currentChildY);
      }
    }
  }
}

// Create a modified sketch for each stage
function createTreeSketch(stageId) {
  return function(p5) {
    let nodes = [];
    let edges = [];
    let treeData = stageData[stageId];

    p5.setup = function() {
      const canvas = p5.createCanvas(p5.windowWidth * 0.4, p5.windowHeight);
      canvas.parent('canvas-container');
      p5.smooth();
      p5.frameRate(60);

      if (treeData) {
        createTreeFromData(treeData);
      }
    };

    p5.windowResized = function() {
      p5.resizeCanvas(p5.windowWidth * 0.4, p5.windowHeight);
    };

    function createTreeFromData(data) {
      // Clear existing nodes and edges
      nodes = [];
      edges = [];

      // Add defensive checks to prevent errors
      if (!data || !data.layout || !Array.isArray(data.layout) || !data.edges || !Array.isArray(data.edges)) {
        console.error("Invalid tree data format:", data);
        return; // Exit if data structure is invalid
      }

      // Find all parent nodes in edges
      const parentNodes = new Set();
      for (const [parentId, childId] of data.edges) {
        parentNodes.add(parentId);
      }

      // Create nodes
      for (let i = 0; i < data.layout.length; i++) {
        const [nx, ny] = data.layout[i];
        // A node is a root if it's a parent and not a child in any edge
        const isRoot = parentNodes.has(i) && data.edges.every(edge => edge[1] !== i);
        const isBestNode = data.is_best_node ? !!data.is_best_node[i] : false;

        const node = new Node(
          nx * p5.width * 0.8 + p5.width * 0.1,
          ny * p5.height * 0.8 + p5.height * 0.1,
          i,
          isRoot,
          isBestNode
        );
        nodes.push(node);
      }

      // If no root was found, make the first parent node visible
      if (!nodes.some(node => node.visible) && parentNodes.size > 0) {
        // Get the first parent node
        const firstParentId = [...parentNodes][0];
        if (nodes[firstParentId]) {
          nodes[firstParentId].visible = true;
        }
      }

      // Make isolated nodes (no edges at all) visible immediately.
      // This handles nodes that lost their none_root parent in the combined view.
      const allEdgeNodes = new Set();
      for (const [p, c] of data.edges) {
        allEdgeNodes.add(p);
        allEdgeNodes.add(c);
      }
      for (let i = 0; i < nodes.length; i++) {
        if (!allEdgeNodes.has(i)) {
          nodes[i].visible = true;
          nodes[i].appearProgress = 1;
        }
      }

      // Create edges
      for (const [parentId, childId] of data.edges) {
        const parent = nodes[parentId];
        const child = nodes[childId];
        if (parent && child) { // Verify both nodes exist
          const isLeft = child.x < parent.x;
          const isRight = child.x > parent.x;
          edges.push(new Edge(parent, child, isLeft, isRight));
        }
      }

      // Select the best node by default, or fall back to first node
      if (nodes.length > 0) {
        const bestIdx = nodes.findIndex(n => n.isBestNode);
        const selectIdx = bestIdx >= 0 ? bestIdx : 0;
        nodes[selectIdx].selected = true;
        updateNodeInfo(selectIdx);
      }
    }

    p5.draw = function() {
      // Use the global background color if available, otherwise use the default bgCol
      const currentBgColor = window.bgColCurrent || bgCol;
      p5.background(currentBgColor);

      // Draw stage separator labels in combined "All" view
      if (treeData && treeData._stage_labels) {
        const stageNames = {
          'Stage_1': 'Stage 1', 'Stage_2': 'Stage 2',
          'Stage_3': 'Stage 3', 'Stage_4': 'Stage 4'
        };
        const seen = new Set();
        for (let i = 0; i < nodes.length; i++) {
          const label = treeData._stage_labels[i];
          if (label && !seen.has(label)) {
            seen.add(label);
            // Draw label at the top-left of this stage's vertical band
            const yPos = nodes[i].y - 25;
            p5.noStroke();
            p5.fill(150);
            p5.textSize(11);
            p5.textAlign(p5.LEFT);
            p5.text(stageNames[label] || label, 10, yPos);
            // Draw a subtle separator line
            p5.stroke(220);
            p5.strokeWeight(0.5);
            p5.line(0, yPos - 10, p5.width, yPos - 10);
          }
        }
      }

      // Update and render edges
      for (const edge of edges) {
        edge.update();
        edge.render(p5);
      }

      // Update and render nodes
      for (const node of nodes) {
        node.update();
        node.render(p5);
      }

      // Handle mouse hover
      p5.cursor(p5.ARROW);
      for (const node of nodes) {
        if (node.isMouseOver(p5)) {
          p5.cursor(p5.HAND);
        }
      }
    };

    p5.mousePressed = function() {
      // Check if any node was clicked
      for (let i = 0; i < nodes.length; i++) {
        if (nodes[i].visible && nodes[i].isMouseOver(p5)) {
          // Deselect all nodes
          nodes.forEach(n => n.selected = false);
          // Select the clicked node
          nodes[i].selected = true;
          // Update the right panel with node info
          updateNodeInfo(i);
          break;
        }
      }
    };

    function updateNodeInfo(nodeIndex) {
      if (treeData) {
        setNodeInfo(
          treeData.code[nodeIndex],
          treeData.plan[nodeIndex],
          treeData.plot_code?.[nodeIndex],
          treeData.plot_plan?.[nodeIndex],
          treeData.metrics?.[nodeIndex],
          treeData.exc_type?.[nodeIndex] || '',
          treeData.exc_info?.[nodeIndex]?.args?.[0] || '',
          treeData.exc_stack?.[nodeIndex] || [],
          treeData.plots?.[nodeIndex] || [],
          treeData.plot_analyses?.[nodeIndex] || [],
          treeData.vlm_feedback_summary?.[nodeIndex] || '',
          treeData.datasets_successfully_tested?.[nodeIndex] || [],
          treeData.exec_time_feedback?.[nodeIndex] || '',
          treeData.exec_time?.[nodeIndex] || '',
          treeData.memory_events?.[nodeIndex] || []
        );
      }
    }
  };
}

// Merge multiple sub-stage tree data objects into one tree for a single main stage.
// Similar to buildCombinedStageData but arranges sub-stages vertically within one stage.
function mergeSubstageTrees(substageDataList) {
  if (!substageDataList || substageDataList.length === 0) return null;
  if (substageDataList.length === 1) return substageDataList[0];

  const nodeFields = [
    'code', 'plan', 'term_out', 'analysis', 'node_id', 'branch_id',
    'inherited_from_node_id', 'exc_type', 'exc_info', 'exc_stack',
    'plots', 'plot_paths', 'plot_analyses', 'vlm_feedback_summary',
    'exec_time', 'exec_time_feedback', 'datasets_successfully_tested',
    'plot_code', 'plot_plan', 'ablation_name', 'hyperparam_name',
    'is_seed_node', 'is_seed_agg_node', 'parse_metrics_plan',
    'parse_metrics_code', 'parse_term_out', 'parse_exc_type',
    'parse_exc_info', 'parse_exc_stack', 'metrics', 'is_best_node',
    'memory_events'
  ];

  const combined = { layout: [], edges: [], _substage_labels: [] };
  for (const f of nodeFields) combined[f] = [];

  let globalIdx = 0;
  const nSubs = substageDataList.length;
  const nodeIdToGlobalIdx = {};

  // First pass: build index maps
  const subIndexMaps = [];
  const subIncluded = [];

  // Track the best node global index per sub-stage for cross-sub-stage linking
  let prevBestGlobalIdx = null;
  // Track none_root children per sub-stage
  const subNoneRootChildren = [];

  for (let si = 0; si < nSubs; si++) {
    const data = substageDataList[si];
    const n = data.layout.length;
    const map = {};
    const included = [];

    const noneRootIdx = data.node_id ? data.node_id.indexOf('none_root') : -1;
    // For sub-stages after the first, skip none_root and link its children to prev best
    const skipNoneRoot = si > 0 && noneRootIdx >= 0;

    // Handle inherited_from_node_id (for the first sub-stage inheriting from another main stage)
    const hasInherited = data.inherited_from_node_id && data.inherited_from_node_id[0];
    const inheritedIdx = hasInherited ? 0 : -1;
    let inheritedParentGlobalIdx = null;
    if (inheritedIdx >= 0) {
      const inheritedFromId = data.inherited_from_node_id[0];
      if (inheritedFromId in nodeIdToGlobalIdx) {
        inheritedParentGlobalIdx = nodeIdToGlobalIdx[inheritedFromId];
      } else {
        for (const [nid, gidx] of Object.entries(nodeIdToGlobalIdx)) {
          if (inheritedFromId.startsWith(nid) || nid.startsWith(inheritedFromId)) {
            inheritedParentGlobalIdx = gidx;
            break;
          }
        }
      }
    }

    // Collect none_root's children for later cross-sub-stage linking
    const noneRootChildLocalIdxs = [];
    if (skipNoneRoot) {
      for (const [pIdx, cIdx] of data.edges) {
        if (pIdx === noneRootIdx) {
          noneRootChildLocalIdxs.push(cIdx);
        }
      }
    }

    for (let i = 0; i < n; i++) {
      if (skipNoneRoot && i === noneRootIdx) continue;
      if (i === inheritedIdx && inheritedParentGlobalIdx !== null) {
        map[i] = inheritedParentGlobalIdx;
        continue;
      }
      map[i] = globalIdx;
      if (data.node_id && data.node_id[i] && data.node_id[i] !== 'none_root') {
        nodeIdToGlobalIdx[data.node_id[i]] = globalIdx;
      }
      included.push(i);
      globalIdx++;
    }

    subIndexMaps.push(map);
    subIncluded.push(included);
    subNoneRootChildren.push(noneRootChildLocalIdxs);
  }

  // Second pass: populate combined arrays and connect sub-stages
  prevBestGlobalIdx = null;

  for (let si = 0; si < nSubs; si++) {
    const data = substageDataList[si];
    const map = subIndexMaps[si];
    const included = subIncluded[si];

    const noneRootIdx = data.node_id ? data.node_id.indexOf('none_root') : -1;
    const skipNoneRoot = si > 0 && noneRootIdx >= 0;

    // Arrange sub-stages in vertical bands
    const bandPad = 0.02;
    const yMin = si / nSubs + bandPad;
    const yMax = (si + 1) / nSubs - bandPad;

    const origYs = included.map(i => data.layout[i][1]);
    const origYMin = Math.min(...origYs);
    const origYMax = Math.max(...origYs);
    const origYRange = origYMax - origYMin || 1;

    for (const i of included) {
      const [ox, oy] = data.layout[i];
      const ny = yMin + ((oy - origYMin) / origYRange) * (yMax - yMin);
      combined.layout.push([ox, ny]);
      combined._substage_labels.push(`Sub-stage ${si + 1}`);

      for (const f of nodeFields) {
        if (data[f] && Array.isArray(data[f]) && i < data[f].length) {
          combined[f].push(data[f][i]);
        } else if (f === 'memory_events') {
          combined[f].push([]);
        } else {
          combined[f].push(null);
        }
      }
    }

    // Add remapped edges, skipping edges from none_root (handled separately below)
    for (const [pIdx, cIdx] of data.edges) {
      if (skipNoneRoot && pIdx === noneRootIdx) continue;
      const gp = map[pIdx];
      const gc = map[cIdx];
      if (gp !== undefined && gc !== undefined && gp !== gc) {
        combined.edges.push([gp, gc]);
      }
    }

    // Connect none_root's children to the previous sub-stage's best node
    if (si > 0 && prevBestGlobalIdx !== null && subNoneRootChildren[si].length > 0) {
      for (const childLocalIdx of subNoneRootChildren[si]) {
        const gc = map[childLocalIdx];
        if (gc !== undefined && gc !== prevBestGlobalIdx) {
          combined.edges.push([prevBestGlobalIdx, gc]);
        }
      }
    }

    // Find this sub-stage's best node for the next sub-stage to link to
    if (data.is_best_node) {
      for (let i = 0; i < data.is_best_node.length; i++) {
        if (data.is_best_node[i] && map[i] !== undefined) {
          prevBestGlobalIdx = map[i];
          break;
        }
      }
    }
  }

  // Copy over stage-level metadata from the last sub-stage
  const last = substageDataList[substageDataList.length - 1];
  combined._stage_labels = combined._substage_labels;
  if (last.completed_stages) combined.completed_stages = last.completed_stages;
  if (last.stage_dir_map) combined.stage_dir_map = last.stage_dir_map;
  if (last.current_stage) combined.current_stage = last.current_stage;

  return combined;
}

// Build a combined tree merging all loaded stages into one connected tree.
// Inherited nodes are merged with the previous stage's best node so that
// the tree is a single connected structure across stages.
function buildCombinedStageData() {
  const stageOrder = ['Stage_1', 'Stage_2', 'Stage_3', 'Stage_4'];
  const loadedStages = stageOrder.filter(s => stageData[s]);

  if (loadedStages.length === 0) return null;
  if (loadedStages.length === 1) return stageData[loadedStages[0]];

  // Per-node array field names to merge
  const nodeFields = [
    'code', 'plan', 'term_out', 'analysis', 'node_id', 'branch_id',
    'inherited_from_node_id', 'exc_type', 'exc_info', 'exc_stack',
    'plots', 'plot_paths', 'plot_analyses', 'vlm_feedback_summary',
    'exec_time', 'exec_time_feedback', 'datasets_successfully_tested',
    'plot_code', 'plot_plan', 'ablation_name', 'hyperparam_name',
    'is_seed_node', 'is_seed_agg_node', 'parse_metrics_plan',
    'parse_metrics_code', 'parse_term_out', 'parse_exc_type',
    'parse_exc_info', 'parse_exc_stack', 'metrics', 'is_best_node',
    'memory_events'
  ];

  const combined = { layout: [], edges: [], _stage_labels: [] };
  for (const f of nodeFields) combined[f] = [];

  let globalIdx = 0;
  const nStages = loadedStages.length;

  // Build a global node_id -> globalIdx map for cross-stage lookups
  const nodeIdToGlobalIdx = {};

  // First pass: build index maps and count included nodes per stage
  const stageIndexMaps = {};
  const stageIncluded = {};

  for (let si = 0; si < nStages; si++) {
    const stageId = loadedStages[si];
    const data = stageData[stageId];
    const n = data.layout.length;
    const map = {};
    const included = [];

    // In the combined view, skip none_root only for stages that have an inherited
    // node (stages 2+), since the inherited node replaces the virtual root's role
    // by connecting to the previous stage's best node. For the first stage (no
    // inheritance), keep none_root so parentless nodes remain connected.
    const hasInherited = data.inherited_from_node_id && data.inherited_from_node_id[0];
    const inheritedIdx = hasInherited ? 0 : -1;
    const noneRootIdx = data.node_id ? data.node_id.indexOf('none_root') : -1;
    const skipNoneRoot = hasInherited && noneRootIdx >= 0;

    // Find the global index of the inherited-from node in a previous stage
    let inheritedParentGlobalIdx = null;
    if (inheritedIdx >= 0) {
      const inheritedFromId = data.inherited_from_node_id[0];
      // Look up by full ID first, then by short prefix match
      if (inheritedFromId in nodeIdToGlobalIdx) {
        inheritedParentGlobalIdx = nodeIdToGlobalIdx[inheritedFromId];
      } else {
        // Try prefix match (node_id in tree may be shorter than inherited_from_node_id)
        for (const [nid, gidx] of Object.entries(nodeIdToGlobalIdx)) {
          if (inheritedFromId.startsWith(nid) || nid.startsWith(inheritedFromId)) {
            inheritedParentGlobalIdx = gidx;
            break;
          }
        }
      }
    }

    for (let i = 0; i < n; i++) {
      if (skipNoneRoot && i === noneRootIdx) continue;
      if (i === inheritedIdx && inheritedParentGlobalIdx !== null) {
        // Map inherited node to the actual inherited-from node in previous stage
        map[i] = inheritedParentGlobalIdx;
        continue;
      }
      map[i] = globalIdx;
      // Register this node's ID in the global map
      if (data.node_id && data.node_id[i] && data.node_id[i] !== 'none_root') {
        nodeIdToGlobalIdx[data.node_id[i]] = globalIdx;
      }
      included.push(i);
      globalIdx++;
    }

    stageIndexMaps[stageId] = map;
    stageIncluded[stageId] = included;
  }

  // Second pass: populate combined arrays
  for (let si = 0; si < nStages; si++) {
    const stageId = loadedStages[si];
    const data = stageData[stageId];
    const map = stageIndexMaps[stageId];
    const included = stageIncluded[stageId];

    // Vertical band for this stage
    const bandPad = 0.02;
    const yMin = si / nStages + bandPad;
    const yMax = (si + 1) / nStages - bandPad;

    // Get y range of included nodes in original layout
    const origYs = included.map(i => data.layout[i][1]);
    const origYMin = Math.min(...origYs);
    const origYMax = Math.max(...origYs);
    const origYRange = origYMax - origYMin || 1;

    for (const i of included) {
      const [ox, oy] = data.layout[i];
      const ny = yMin + ((oy - origYMin) / origYRange) * (yMax - yMin);
      combined.layout.push([ox, ny]);
      combined._stage_labels.push(stageId);

      for (const f of nodeFields) {
        if (data[f] && Array.isArray(data[f]) && i < data[f].length) {
          combined[f].push(data[f][i]);
        } else if (f === 'memory_events') {
          combined[f].push([]);
        } else {
          combined[f].push(null);
        }
      }
    }

    // Add edges (remapped to global indices)
    for (const [pIdx, cIdx] of data.edges) {
      const gp = map[pIdx];
      const gc = map[cIdx];
      if (gp !== undefined && gc !== undefined && gp !== gc) {
        combined.edges.push([gp, gc]);
      }
    }
  }

  return combined;
}

// Start a new p5 sketch for the given stage
function startSketch(stageId) {
  if (currentSketch) {
    currentSketch.remove();
  }

  if (stageId === 'All') {
    const combined = buildCombinedStageData();
    if (combined) {
      stageData['All'] = combined;
      currentSketch = new p5(createTreeSketch('All'));
      document.getElementById('stage-info').innerHTML =
        `<strong>All Stages (Combined Tree)</strong>`;
    }
  } else if (stageData[stageId]) {
    currentSketch = new p5(createTreeSketch(stageId));

    // Update stage info
    const stageNumber = stageId.split('_')[1];
    let stageDesc = '';
    switch(stageId) {
      case 'Stage_1': stageDesc = 'Preliminary Investigation'; break;
      case 'Stage_2': stageDesc = 'Baseline Tuning'; break;
      case 'Stage_3': stageDesc = 'Research Agenda Execution'; break;
      case 'Stage_4': stageDesc = 'Ablation Studies'; break;
    }

    document.getElementById('stage-info').innerHTML =
      `<strong>Current Stage: ${stageNumber} - ${stageDesc}</strong>`;
  }
}

// Handle tab selection
function selectStage(stageId) {
  if (stageId !== 'All' && (!stageData[stageId] || !availableStages.includes(stageId))) {
    return; // Don't allow selection of unavailable stages
  }

  // Update active tab styles
  document.querySelectorAll('.tab').forEach(tab => {
    tab.classList.remove('active');
  });
  document.querySelector(`.tab[data-stage="${stageId}"]`).classList.add('active');

  // Start the new sketch
  currentStage = stageId;
  startSketch(stageId);
}

// Function to load the tree data for all stages
async function loadAllStageData(baseTreeData) {
  console.log("Loading stage data with base data:", baseTreeData);

  // The base tree data is for the current stage
  const currentStageId = baseTreeData.current_stage || 'Stage_1';

  // Ensure base tree data is valid and has required properties
  if (baseTreeData && Array.isArray(baseTreeData.layout) && Array.isArray(baseTreeData.edges)) {
    stageData[currentStageId] = baseTreeData;
    addAvailableStage(currentStageId);
    console.log(`Added current stage ${currentStageId} to available stages`);
  } else {
    console.warn(`Current stage ${currentStageId} data is invalid:`, baseTreeData);
  }

  // Use relative path to load other stage trees
  const logDirPath = baseTreeData.log_dir_path || '.';
  console.log("Log directory path:", logDirPath);

  // Load data for each stage if available
  const stageNames = ['Stage_1', 'Stage_2', 'Stage_3', 'Stage_4'];
  const stageDirMap = baseTreeData.stage_dir_map || defaultStageDirMap;
  const substageDirMap = baseTreeData.substage_dir_map || {};

  for (const stage of stageNames) {

    if (baseTreeData.completed_stages && baseTreeData.completed_stages.includes(stage)) {
      if (stageData[stage]) {
        continue;
      }

      // Check if this stage has multiple sub-stages
      const substageDirs = substageDirMap[stage];
      if (substageDirs && substageDirs.length > 1) {
        console.log(`Stage ${stage} has ${substageDirs.length} sub-stages, loading all...`);
        const substageDataList = [];
        for (const subDir of substageDirs) {
          try {
            const url = `${logDirPath}/${subDir}/tree_data.json`;
            console.log(`Loading sub-stage from ${url}`);
            const response = await fetch(url);
            if (response.ok) {
              const data = await response.json();
              if (data && Array.isArray(data.layout) && Array.isArray(data.edges)) {
                substageDataList.push(data);
              }
            }
          } catch (error) {
            console.error(`Error loading sub-stage ${subDir}:`, error);
          }
        }
        if (substageDataList.length > 0) {
          stageData[stage] = mergeSubstageTrees(substageDataList);
          addAvailableStage(stage);
          console.log(`Merged ${substageDataList.length} sub-stages for ${stage}`);
        }
        continue;
      }

      try {
        const stageDirName = stageDirMap[stage] || stage.toLowerCase();
        if (!stageDirName) {
          console.warn(`No stage directory mapping for ${stage}`);
          continue;
        }
        console.log(`Attempting to load data for ${stage} from ${logDirPath}/${stageDirName}/tree_data.json`);
        const response = await fetch(`${logDirPath}/${stageDirName}/tree_data.json`);

        if (response.ok) {
          const data = await response.json();

          // Validate the loaded data
          if (data && Array.isArray(data.layout) && Array.isArray(data.edges)) {
            stageData[stage] = data;
            addAvailableStage(stage);
            console.log(`Successfully loaded and validated data for ${stage}`);
          } else {
            console.warn(`Loaded data for ${stage} is invalid:`, data);
          }
        } else {
          console.warn(`Failed to load data for ${stage} - HTTP status ${response.status}`);
        }
      } catch (error) {
        console.error(`Error loading data for ${stage}:`, error);
      }
    } else {
      console.log(`Skipping stage ${stage} - not in completed stages list:`, baseTreeData.completed_stages);
    }
  }

  // Update tab visibility based on available stages
  updateTabVisibility();

  // Start with the combined "All" view if multiple stages are available,
  // otherwise show the single available stage
  if (availableStages.length > 1) {
    selectStage('All');
  } else if (availableStages.length > 0) {
    selectStage(availableStages[0]);
  } else {
    console.warn("No stages available to display");
    // Display a message in the canvas area
    document.getElementById('canvas-container').innerHTML =
      '<div style="padding: 20px; color: #333; text-align: center;"><h3>No valid tree data available to display</h3></div>';
  }
}

// Update tab visibility based on available stages
function updateTabVisibility() {
  const tabs = document.querySelectorAll('.tab');
  tabs.forEach(tab => {
    const stageId = tab.getAttribute('data-stage');
    if (stageId === 'All') {
      // "All" tab is enabled when more than one stage is available
      if (availableStages.length > 1) {
        tab.classList.remove('disabled');
      } else {
        tab.classList.add('disabled');
      }
    } else if (availableStages.includes(stageId)) {
      tab.classList.remove('disabled');
    } else {
      tab.classList.add('disabled');
    }
  });
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function getEventCategory(op) {
  return MEMORY_OP_CATEGORIES[op] || 'other';
}

// Infer phase from operation type when phase is null
function inferPhaseFromOp(op) {
  if (!op) return 'system';
  // Node management operations (fork, branch creation)
  if (op.includes('node_fork') || op.includes('branch')) {
    return 'node_setup';
  }
  // Resource operations
  if (op.includes('resources') || op.includes('resource')) {
    return 'resource_init';
  }
  // Initial core setup operations
  if (op.includes('core_set') || op.includes('set_core')) {
    return 'initialization';
  }
  if (op.includes('core_get') || op.includes('get_core')) {
    return 'initialization';
  }
  // Archival operations without explicit phase
  if (op.includes('archival')) {
    return 'archival_ops';
  }
  return 'system';
}

function groupMemoryEvents(events) {
  const grouped = {};
  if (!Array.isArray(events)) {
    return grouped;
  }
  for (const event of events) {
    if (!event || typeof event !== 'object') {
      continue;
    }
    // Use explicit phase if available, otherwise infer from operation type
    const phase = event.phase || inferPhaseFromOp(event.op);
    if (!grouped[phase]) {
      grouped[phase] = [];
    }
    grouped[phase].push(event);
  }
  return grouped;
}

// Calculate memory operation statistics for a phase
function calculateMemoryStats(events) {
  const stats = {
    total: 0,
    reads: 0,
    writes: 0,
    deletes: 0,
    forks: 0,
    recalls: 0,
    resources: 0,
    other: 0,
  };
  if (!Array.isArray(events)) {
    return stats;
  }
  for (const event of events) {
    if (!event || typeof event !== 'object') continue;
    const category = getEventCategory(event.op);
    stats.total++;
    stats[category] = (stats[category] || 0) + 1;
  }
  return stats;
}

// Filter events by category
function filterEventsByCategory(events, filter) {
  if (filter === 'all') return events;
  return events.filter(event => {
    const category = getEventCategory(event.op);
    return category === filter;
  });
}

function sortMemoryPhases(phases) {
  // Order: system phases first, then explicit phases, then other inferred phases
  const order = [
    'node_setup',      // Node/branch creation
    'resource_init',   // Resource initialization
    'initialization',  // Core key-value setup
    'phase0',          // Explicit phases
    'phase1',
    'phase2',
    'phase3',
    'phase4',
    'define_metrics',
    'journal_summary',
    'archival_ops',    // Archival operations
    'system',          // Other system operations
  ];
  return phases.slice().sort((a, b) => {
    const aIndex = order.indexOf(a);
    const bIndex = order.indexOf(b);
    if (aIndex !== -1 || bIndex !== -1) {
      return (aIndex === -1 ? 99 : aIndex) - (bIndex === -1 ? 99 : bIndex);
    }
    return String(a).localeCompare(String(b));
  });
}

function formatMemoryEvent(event) {
  const op = event.op || 'memory_event';
  const memType = event.memory_type || 'unknown';
  const category = getEventCategory(op);
  const config = CATEGORY_CONFIG[category] || CATEGORY_CONFIG['other'];
  const ts = event.ts ? new Date(event.ts * 1000).toLocaleString() : '';
  const metaParts = [];
  if (ts) metaParts.push(ts);
  if (event.node_id) metaParts.push(`node_id=${event.node_id}`);
  if (event.branch_id) metaParts.push(`branch_id=${event.branch_id}`);
  const metaText = metaParts.join(' | ');
  const details = event.details ? JSON.stringify(event.details, null, 2) : '';
  const detailsHtml = details ? `<pre class="memory-event-details">${escapeHtml(details)}</pre>` : '';
  // Extract key info for display
  let keyInfoHtml = '';
  if (event.details) {
    const d = event.details;
    if (d.key) keyInfoHtml += `<span class="memory-key">key: ${escapeHtml(d.key)}</span> `;
    if (d.value_chars) keyInfoHtml += `<span class="memory-size">${d.value_chars} chars</span> `;
    if (d.record_id) keyInfoHtml += `<span class="memory-id">record: ${d.record_id}</span> `;
  }
  return `
    <div class="memory-event" data-category="${category}">
      <div class="memory-event-header">
        <span class="memory-badge" style="background-color: ${config.color}">${config.icon} ${config.label}</span>
        <span class="memory-op">${escapeHtml(op)}</span>
        <span class="memory-type">(${escapeHtml(memType)})</span>
      </div>
      ${keyInfoHtml ? `<div class="memory-event-keyinfo">${keyInfoHtml}</div>` : ''}
      <div class="memory-event-meta">${escapeHtml(metaText)}</div>
      ${detailsHtml}
    </div>
  `;
}

// Render memory summary table
function renderMemorySummary(stats) {
  const categories = ['reads', 'writes', 'deletes', 'forks', 'recalls', 'resources', 'maintenance'];
  let rows = '';
  for (const cat of categories) {
    const config = CATEGORY_CONFIG[cat];
    const count = stats[cat] || 0;
    if (count > 0) {
      rows += `<tr>
        <td><span class="memory-badge" style="background-color: ${config.color}">${config.icon} ${config.label}</span></td>
        <td class="memory-count">${count}</td>
      </tr>`;
    }
  }
  if (!rows) {
    return '<p class="memory-no-events">No memory events</p>';
  }
  return `
    <table class="memory-summary-table">
      <thead><tr><th>Operation</th><th>Count</th></tr></thead>
      <tbody>${rows}</tbody>
      <tfoot><tr><th>Total</th><th>${stats.total}</th></tr></tfoot>
    </table>
  `;
}

// Handle filter button clicks
function setMemoryFilter(filter) {
  memoryCurrentFilter = filter;
  const buttons = document.querySelectorAll('.memory-filter');
  buttons.forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-filter') === filter);
  });
  renderMemoryPhase();
}

function renderMemoryPhase() {
  const labelElm = document.getElementById('memory-phase-label');
  const contentElm = document.getElementById('memory-content');
  const summaryElm = document.getElementById('memory-summary');
  const prevBtn = document.getElementById('memory-prev');
  const nextBtn = document.getElementById('memory-next');
  if (!labelElm || !contentElm || !prevBtn || !nextBtn) {
    return;
  }
  if (!memoryPhaseKeys.length) {
    labelElm.textContent = 'No memory events';
    if (summaryElm) summaryElm.innerHTML = '';
    contentElm.innerHTML = '<p>No memory events recorded for this node.</p>';
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    return;
  }
  const phase = memoryPhaseKeys[memoryPhaseIndex] || 'unknown';
  labelElm.textContent = phase;
  const allEvents = memoryEventsByPhase[phase] || [];
  const stats = calculateMemoryStats(allEvents);
  if (summaryElm) {
    summaryElm.innerHTML = renderMemorySummary(stats);
  }
  const filteredEvents = filterEventsByCategory(allEvents, memoryCurrentFilter);
  if (filteredEvents.length === 0 && memoryCurrentFilter !== 'all') {
    contentElm.innerHTML = `<p>No ${memoryCurrentFilter} events in this phase. <a href="#" onclick="setMemoryFilter('all'); return false;">Show all</a></p>`;
  } else {
    contentElm.innerHTML = filteredEvents.map(formatMemoryEvent).join('');
  }
  const disableNav = memoryPhaseKeys.length <= 1;
  prevBtn.disabled = disableNav;
  nextBtn.disabled = disableNav;
}

function shiftMemoryPhase(direction) {
  if (!memoryPhaseKeys.length) {
    return;
  }
  memoryPhaseIndex = (memoryPhaseIndex + direction + memoryPhaseKeys.length) % memoryPhaseKeys.length;
  renderMemoryPhase();
}

function updateMemoryPanel(events) {
  memoryEventsByPhase = groupMemoryEvents(events);
  memoryPhaseKeys = sortMemoryPhases(Object.keys(memoryEventsByPhase));
  memoryPhaseIndex = 0;
  memoryCurrentFilter = 'all';
  // Reset filter buttons
  const buttons = document.querySelectorAll('.memory-filter');
  buttons.forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-filter') === 'all');
  });
  renderMemoryPhase();
}

// Utility function to set the node info in the right panel
const setNodeInfo = (code, plan, plot_code, plot_plan, metrics = null, exc_type = '', exc_info = '',
    exc_stack = [], plots = [], plot_analyses = [], vlm_feedback_summary = '',
    datasets_successfully_tested = [], exec_time_feedback = '', exec_time = '', memory_events = []) => {
  const codeElm = document.getElementById("code");
  if (codeElm) {
    if (code) {
      codeElm.innerHTML = hljs.highlight(code, { language: "python" }).value;
    } else {
      codeElm.innerHTML = '<p>No code available</p>';
    }
  }

  const planElm = document.getElementById("plan");
  if (planElm) {
    if (plan) {
      planElm.innerHTML = hljs.highlight(plan, { language: "plaintext" }).value;
    } else {
      planElm.innerHTML = '<p>No plan available</p>';
    }
  }

  const plot_codeElm = document.getElementById("plot_code");
  if (plot_codeElm) {
    if (plot_code) {
      plot_codeElm.innerHTML = hljs.highlight(plot_code, { language: "python" }).value;
    } else {
      plot_codeElm.innerHTML = '<p>No plot code available</p>';
    }
  }

  const plot_planElm = document.getElementById("plot_plan");
  if (plot_planElm) {
    if (plot_plan) {
      plot_planElm.innerHTML = hljs.highlight(plot_plan, { language: "plaintext" }).value;
    } else {
      plot_planElm.innerHTML = '<p>No plot plan available</p>';
    }
  }

  const metricsElm = document.getElementById("metrics");
  if (metricsElm) {
      let metricsContent = `<h3>Metrics:</h3>`;
      if (metrics && metrics.metric_names) {
          for (const metric of metrics.metric_names) {
              metricsContent += `<div class="metric-group">`;
              metricsContent += `<h4>${metric.metric_name}</h4>`;
              metricsContent += `<p><strong>Description:</strong> ${metric.description || 'N/A'}</p>`;
              metricsContent += `<p><strong>Optimization:</strong> ${metric.lower_is_better ? 'Minimize' : 'Maximize'}</p>`;

              // Create table for dataset values
              metricsContent += `<table class="metric-table">
                  <tr>
                      <th>Dataset</th>
                      <th>Value</th>
                  </tr>`;

              for (const dataPoint of metric.data) {
                  metricsContent += `<tr>
                      <td>${dataPoint.dataset_name}</td>
                      <td>${dataPoint.value?.toFixed(4) || 'N/A'}</td>
                  </tr>`;
              }

              metricsContent += `</table></div>`;
          }
      } else if (metrics === null) {
          metricsContent += `<p>No metrics available</p>`;
      }
      metricsElm.innerHTML = metricsContent;
  }

  // Add plots display
  const plotsElm = document.getElementById("plots");
  if (plotsElm) {
      if (plots && plots.length > 0) {
          let plotsContent = '';
          plots.forEach(plotPath => {
              plotsContent += `
                  <div class="plot-item">
                      <img src="${plotPath}" alt="Experiment Plot" onerror="console.error('Failed to load plot:', this.src)"/>
                  </div>`;
          });
          plotsElm.innerHTML = plotsContent;
      } else {
          plotsElm.innerHTML = '';
      }
  }

  // Add error info display
  const errorElm = document.getElementById("exc_info");
  if (errorElm) {
    if (exc_type) {
      let errorContent = `<h3 style="color: #ff5555">Exception Information:</h3>
                          <p><strong>Type:</strong> ${exc_type}</p>`;

      if (exc_info) {
        errorContent += `<p><strong>Details:</strong> <pre>${JSON.stringify(exc_info, null, 2)}</pre></p>`;
      }

      if (exc_stack) {
        errorContent += `<p><strong>Stack Trace:</strong> <pre>${exc_stack.join('\n')}</pre></p>`;
      }

      errorElm.innerHTML = errorContent;
    } else {
      errorElm.innerHTML = "No exception info available";
    }
  }

  const exec_timeElm = document.getElementById("exec_time");
  if (exec_timeElm) {
    let exec_timeContent = '<div id="exec_time"><h3>Execution Time (in seconds):</h3><p>' + exec_time + '</p></div>';
    exec_timeElm.innerHTML = exec_timeContent;
  }

  const exec_time_feedbackElm = document.getElementById("exec_time_feedback");
  if (exec_time_feedbackElm) {
    let exec_time_feedbackContent = '<div id="exec_time_feedback_content">'
    exec_time_feedbackContent += '<h3>Execution Time Feedback:</h3>'
    exec_time_feedbackContent += '<p>' + exec_time_feedback + '</p>'
    exec_time_feedbackContent += '</div>';
    exec_time_feedbackElm.innerHTML = exec_time_feedbackContent;
  }

  const vlm_feedbackElm = document.getElementById("vlm_feedback");
  if (vlm_feedbackElm) {
      let vlm_feedbackContent = '';

      if (plot_analyses && plot_analyses.length > 0) {
          vlm_feedbackContent += `<h3>Plot Analysis:</h3>`;
          plot_analyses.forEach(analysis => {
              if (analysis && analysis.plot_path) {  // Add null check
                  vlm_feedbackContent += `
                      <div class="plot-analysis">
                          <h4>Analysis for ${analysis.plot_path.split('/').pop()}</h4>
                          <p>${analysis.analysis || 'No analysis available'}</p>
                          <ul class="key-findings">
                              ${(analysis.key_findings || []).map(finding => `<li>${finding}</li>`).join('')}
                          </ul>
                      </div>`;
              } else {
                  console.warn('Received invalid plot analysis:', analysis);
                  vlm_feedbackContent += `
                      <div class="plot-analysis">
                          <p>Invalid plot analysis data received</p>
                      </div>`;
              }
          });
      }

      // Add actionable insights if available
      if (vlm_feedback_summary && typeof vlm_feedback_summary === 'string') {
          vlm_feedbackContent += `
              <div class="vlm_feedback">
                  <h3>VLM Feedback Summary:</h3>
                  <p>${vlm_feedback_summary}</p>
              </div>`;
      }

      console.log("Datasets successfully tested:", datasets_successfully_tested);
      if (datasets_successfully_tested && datasets_successfully_tested.length > 0) {
          vlm_feedbackContent += `
              <div id="datasets_successfully_tested">
                  <h3>Datasets Successfully Tested:</h3>
                  <p>${datasets_successfully_tested.join(', ')}</p>
              </div>`;
      }

      if (!vlm_feedbackContent) {
          vlm_feedbackContent = '<p>No insights available for this experiment.</p>';
      }

      vlm_feedbackElm.innerHTML = vlm_feedbackContent;
  }

  const datasets_successfully_testedElm = document.getElementById("datasets_successfully_tested");
  if (datasets_successfully_testedElm) {
      let datasets_successfully_testedContent = '';
      if (datasets_successfully_tested && datasets_successfully_tested.length > 0) {
          datasets_successfully_testedContent = `<h3>Datasets Successfully Tested:</h3><ul>`;
          datasets_successfully_tested.forEach(dataset => {
              datasets_successfully_testedContent += `<li>${dataset}</li>`;
          });
          datasets_successfully_testedContent += `</ul>`;
      } else {
          datasets_successfully_testedContent = '<p>No datasets tested yet</p>';
      }
      datasets_successfully_testedElm.innerHTML = datasets_successfully_testedContent;
  }

  updateMemoryPanel(memory_events);
};

// Initialize with the provided tree data
const treeStructData = "PLACEHOLDER_TREE_DATA";

if (!treeStructData.log_dir_path) {
  treeStructData.log_dir_path = inferLogDirPath(window.location.pathname);
}

const inferredStage = inferStageIdFromPath(window.location.pathname);
if (!treeStructData.current_stage) {
  treeStructData.current_stage = inferredStage || 'Stage_1';
} else if (!treeStructData.current_stage.startsWith('Stage_') && inferredStage) {
  treeStructData.current_stage = inferredStage;
}

// Initialize background color
window.bgColCurrent = bgCol;

// Function to set background color that can be called from the console
function setBackgroundColor(color) {
  // Update the global color
  updateBackgroundColor(color);

  // Refresh the current sketch to apply the new background color
  if (currentStage) {
    startSketch(currentStage);
  }
}

// Load all stage data and initialize the visualization
loadAllStageData(treeStructData);
