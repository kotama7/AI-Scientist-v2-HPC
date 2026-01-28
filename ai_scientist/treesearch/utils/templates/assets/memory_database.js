/**
 * Memory Database Viewer - Core JavaScript
 * Handles tree visualization and memory data display
 */

// ===== Constants =====

// Memory operation type classification (based on docs/memory.md and actual logs)
const MEMORY_OP_TYPES = {
    // Prompt Injection
    'render_for_prompt': { type: 'read', label: 'Memory Injection', category: 'injection' },

    // Core Memory
    'mem_core_get': { type: 'read', label: 'Core Get', category: 'core' },
    'mem_core_set': { type: 'write', label: 'Core Set', category: 'core' },
    'mem_core_del': { type: 'write', label: 'Core Delete', category: 'core' },
    'set_core': { type: 'write', label: 'Core Set', category: 'core' },
    'get_core': { type: 'read', label: 'Core Get', category: 'core' },
    'core_evict': { type: 'write', label: 'Core Evict', category: 'core' },
    'ingest_idea_md': { type: 'write', label: 'Ingest Idea MD', category: 'core' },

    // Recall Memory
    'mem_recall_append': { type: 'write', label: 'Recall Append', category: 'recall' },
    'mem_recall_search': { type: 'read', label: 'Recall Search', category: 'recall' },
    'recall_evict': { type: 'write', label: 'Recall Evict', category: 'recall' },
    'recall_summarize': { type: 'write', label: 'Recall Summarize', category: 'recall' },

    // Archival Memory
    'mem_archival_write': { type: 'write', label: 'Archival Write', category: 'archival' },
    'mem_archival_update': { type: 'write', label: 'Archival Update', category: 'archival' },
    'mem_archival_search': { type: 'read', label: 'Archival Search', category: 'archival' },
    'mem_archival_get': { type: 'read', label: 'Archival Get', category: 'archival' },
    'write_archival': { type: 'write', label: 'Archival Write', category: 'archival' },
    'retrieve_archival': { type: 'read', label: 'Archival Retrieve', category: 'archival' },

    // Node Operations
    'mem_node_fork': { type: 'fork', label: 'Node Fork', category: 'node' },
    'mem_node_read': { type: 'read', label: 'Node Read', category: 'node' },
    'mem_node_write': { type: 'write', label: 'Node Write', category: 'node' },

    // Resource Operations
    'mem_resources_index_update': { type: 'system', label: 'Resources Index Update', category: 'resources' },
    'mem_resources_snapshot_upsert': { type: 'system', label: 'Resources Snapshot', category: 'resources' },

    // Memory Management
    'check_memory_pressure': { type: 'system', label: 'Pressure Check', category: 'management' },
    'consolidate': { type: 'system', label: 'Consolidation', category: 'management' },
    'importance_evaluation': { type: 'system', label: 'Importance Eval', category: 'management' },

    // LLM Memory Operations
    'llm_core_set': { type: 'llm', label: 'LLM Core Set', category: 'llm' },
    'llm_core_get': { type: 'llm', label: 'LLM Core Get', category: 'llm' },
    'llm_core_delete': { type: 'llm', label: 'LLM Core Delete', category: 'llm' },
    'llm_archival_write': { type: 'llm', label: 'LLM Archival Write', category: 'llm' },
    'llm_archival_search': { type: 'llm', label: 'LLM Archival Search', category: 'llm' },
    'llm_archival_update': { type: 'llm', label: 'LLM Archival Update', category: 'llm' },
    'llm_recall_append': { type: 'llm', label: 'LLM Recall Append', category: 'llm' },
    'llm_recall_search': { type: 'llm', label: 'LLM Recall Search', category: 'llm' },
    'llm_recall_evict': { type: 'llm', label: 'LLM Recall Evict', category: 'llm' },
    'llm_recall_summarize': { type: 'llm', label: 'LLM Recall Summarize', category: 'llm' },
    'llm_consolidate': { type: 'llm', label: 'LLM Consolidate', category: 'llm' },
};

// Phase display names
const PHASE_LABELS = {
    'phase0': 'Phase 0: Planning',
    'phase1': 'Phase 1: Download/Install',
    'phase2': 'Phase 2: Implementation',
    'phase3': 'Phase 3: Evaluation',
    'phase4': 'Phase 4: Analysis',
    'summary': 'Summary',
};

// ===== Utility Functions =====

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

function formatTimestamp(ts) {
    if (!ts) return '';
    try {
        return new Date(ts * 1000).toLocaleTimeString();
    } catch {
        return '';
    }
}

function truncateText(text, maxLen = 100) {
    if (!text) return '';
    const str = String(text);
    return str.length > maxLen ? str.substring(0, maxLen) + '...' : str;
}

// ===== Memory Call Rendering =====

// Unique ID counter for collapsible details
let detailIdCounter = 0;

function getOpInfo(op) {
    return MEMORY_OP_TYPES[op] || { type: 'other', label: op, category: 'other' };
}

function formatJsonValue(value, indent = 0) {
    if (value === null) return '<span class="json-null">null</span>';
    if (typeof value === 'boolean') return `<span class="json-boolean">${value}</span>`;
    if (typeof value === 'number') return `<span class="json-number">${value}</span>`;
    if (typeof value === 'string') {
        const escaped = escapeHtml(value);
        if (value.length > 500) {
            return `<span class="json-string">"${escaped.substring(0, 500)}..."</span>`;
        }
        return `<span class="json-string">"${escaped}"</span>`;
    }
    if (Array.isArray(value)) {
        if (value.length === 0) return '[]';
        const items = value.map(v => formatJsonValue(v, indent + 2)).join(', ');
        return `[${items}]`;
    }
    if (typeof value === 'object') {
        const entries = Object.entries(value);
        if (entries.length === 0) return '{}';
        const formatted = entries.map(([k, v]) =>
            `<span class="json-key">"${escapeHtml(k)}"</span>: ${formatJsonValue(v, indent + 2)}`
        ).join(',\n' + ' '.repeat(indent + 2));
        return `{\n${' '.repeat(indent + 2)}${formatted}\n${' '.repeat(indent)}}`;
    }
    return escapeHtml(String(value));
}

function renderFullDetails(item) {
    const detailId = `detail-${detailIdCounter++}`;
    const details = item.details || {};
    const hasDetails = Object.keys(details).length > 0;

    if (!hasDetails) return '';

    // Build sections for different types of content
    let sections = '';

    // Value preview (for core/archival operations)
    if (details.value_preview) {
        sections += `
            <div class="full-details-section">
                <div class="full-details-label">Value Content</div>
                <div class="full-details-content">${escapeHtml(details.value_preview)}</div>
            </div>`;
    }

    // Text preview (for archival/recall)
    if (details.text_preview) {
        sections += `
            <div class="full-details-section">
                <div class="full-details-label">Text Content</div>
                <div class="full-details-content">${escapeHtml(details.text_preview)}</div>
            </div>`;
    }

    // Summary preview
    if (details.summary_preview && !details.text_preview) {
        sections += `
            <div class="full-details-section">
                <div class="full-details-label">Summary</div>
                <div class="full-details-content">${escapeHtml(details.summary_preview)}</div>
            </div>`;
    }

    // Tags
    if (details.tags && details.tags.length > 0) {
        sections += `
            <div class="full-details-section">
                <div class="full-details-label">Tags</div>
                <div class="full-details-content">${details.tags.map(t => escapeHtml(t)).join('\n')}</div>
            </div>`;
    }

    // Full JSON details
    sections += `
        <div class="full-details-section">
            <div class="full-details-label">All Details (JSON)</div>
            <div class="full-details-content full-details-json"><pre>${formatJsonValue(details)}</pre></div>
        </div>`;

    // Raw item data
    const rawData = { ...item };
    delete rawData.details; // Already shown above
    sections += `
        <div class="full-details-section">
            <div class="full-details-label">Metadata</div>
            <div class="full-details-content full-details-json"><pre>${formatJsonValue(rawData)}</pre></div>
        </div>`;

    return `
        <button class="detail-toggle-btn" onclick="toggleFullDetails('${detailId}', this)">
            <span class="toggle-icon">▼</span>
            <span>Show Full Details</span>
        </button>
        <div class="full-details" id="${detailId}">
            ${sections}
        </div>`;
}

function toggleFullDetails(detailId, btn) {
    const detailsEl = document.getElementById(detailId);
    if (!detailsEl) return;

    const isExpanded = detailsEl.classList.toggle('expanded');
    btn.classList.toggle('expanded', isExpanded);
    btn.querySelector('span:last-child').textContent = isExpanded ? 'Hide Details' : 'Show Full Details';
}

function renderMemoryCallDetails(item) {
    const op = item.op || 'unknown';
    const details = item.details || {};
    const category = getOpInfo(op).category;

    let quickSummary = '';

    if (op === 'render_for_prompt') {
        quickSummary = `
            <div class="memory-call-details">
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Budget</span>
                    <span class="memory-call-detail-value">${(details.budget_chars || 0).toLocaleString()} chars</span>
                </div>
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Core items</span>
                    <span class="memory-call-detail-value">${details.core_count || 0}</span>
                </div>
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Recall items</span>
                    <span class="memory-call-detail-value">${details.recall_count || 0}</span>
                </div>
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Archival items</span>
                    <span class="memory-call-detail-value">${details.archival_count || 0}</span>
                </div>
                ${details.resource_items ? `
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Resources</span>
                    <span class="memory-call-detail-value">${details.resource_items}</span>
                </div>` : ''}
            </div>`;
    } else if (op === 'mem_recall_append') {
        quickSummary = `
            <div class="memory-call-details">
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Kind</span>
                    <span class="memory-call-detail-value">${escapeHtml(details.kind || 'N/A')}</span>
                </div>
                ${details.summary_preview ? `
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Summary</span>
                    <span class="memory-call-detail-value">${escapeHtml(truncateText(details.summary_preview, 100))}</span>
                </div>` : ''}
            </div>`;
    } else if (op === 'mem_node_fork') {
        quickSummary = `
            <div class="memory-call-details">
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Parent</span>
                    <span class="memory-call-detail-value">${details.parent_node_id ? truncateText(details.parent_node_id, 16) : 'None (root)'}</span>
                </div>
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Child Branch</span>
                    <span class="memory-call-detail-value">${truncateText(details.child_branch_id || 'N/A', 16)}</span>
                </div>
            </div>`;
    } else if (op === 'check_memory_pressure') {
        quickSummary = `
            <div class="memory-call-details">
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Pressure</span>
                    <span class="memory-call-detail-value">${escapeHtml(details.pressure_level || details.pressure || 'N/A')}</span>
                </div>
                ${details.usage_percent !== undefined ? `
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Usage</span>
                    <span class="memory-call-detail-value">${(details.usage_percent * 100).toFixed(1)}%</span>
                </div>` : ''}
            </div>`;
    } else if (op === 'mem_archival_write' || op === 'write_archival') {
        quickSummary = `
            <div class="memory-call-details">
                ${details.record_id ? `
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Record ID</span>
                    <span class="memory-call-detail-value">${details.record_id}</span>
                </div>` : ''}
                ${details.text_preview ? `
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Content</span>
                    <span class="memory-call-detail-value">${escapeHtml(truncateText(details.text_preview, 150))}</span>
                </div>` : ''}
                ${details.text_chars ? `
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Size</span>
                    <span class="memory-call-detail-value">${details.text_chars.toLocaleString()} chars</span>
                </div>` : ''}
            </div>`;
    } else if (op === 'mem_core_set' || op === 'set_core' || op === 'ingest_idea_md') {
        quickSummary = `
            <div class="memory-call-details">
                ${details.key ? `
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Key</span>
                    <span class="memory-call-detail-value">${escapeHtml(details.key)}</span>
                </div>` : ''}
                ${details.value_preview ? `
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Value</span>
                    <span class="memory-call-detail-value">${escapeHtml(truncateText(details.value_preview, 150))}</span>
                </div>` : ''}
                ${details.value_chars ? `
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Size</span>
                    <span class="memory-call-detail-value">${details.value_chars.toLocaleString()} chars</span>
                </div>` : ''}
                ${details.importance !== undefined ? `
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Importance</span>
                    <span class="memory-call-detail-value">${details.importance}</span>
                </div>` : ''}
            </div>`;
    } else if (op === 'get_core' || op === 'mem_core_get') {
        quickSummary = `
            <div class="memory-call-details">
                ${details.key ? `
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Key</span>
                    <span class="memory-call-detail-value">${escapeHtml(details.key)}</span>
                </div>` : ''}
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Found</span>
                    <span class="memory-call-detail-value">${details.found ? 'Yes' : 'No'}</span>
                </div>
            </div>`;
    } else if (op === 'core_evict') {
        quickSummary = `
            <div class="memory-call-details">
                ${details.key ? `
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Key</span>
                    <span class="memory-call-detail-value">${escapeHtml(details.key)}</span>
                </div>` : ''}
                ${details.reason ? `
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Reason</span>
                    <span class="memory-call-detail-value">${escapeHtml(details.reason)}</span>
                </div>` : ''}
            </div>`;
    } else if (category === 'archival' && (op.includes('search') || op.includes('get'))) {
        quickSummary = `
            <div class="memory-call-details">
                ${details.query ? `
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Query</span>
                    <span class="memory-call-detail-value">${escapeHtml(truncateText(details.query, 100))}</span>
                </div>` : ''}
                ${details.k || details.result_count ? `
                <div class="memory-call-detail-row">
                    <span class="memory-call-detail-key">Results</span>
                    <span class="memory-call-detail-value">${details.k || details.result_count}</span>
                </div>` : ''}
            </div>`;
    } else if (Object.keys(details).length > 0) {
        // Generic fallback - show first few key details
        const rows = Object.entries(details).slice(0, 4).map(([k, v]) => `
            <div class="memory-call-detail-row">
                <span class="memory-call-detail-key">${escapeHtml(k)}</span>
                <span class="memory-call-detail-value">${escapeHtml(truncateText(String(v), 80))}</span>
            </div>`).join('');
        quickSummary = `<div class="memory-call-details">${rows}</div>`;
    }

    // Add the expandable full details button
    return quickSummary + renderFullDetails(item);
}

function renderMemoryCall(item, isInherited = false) {
    const op = item.op || 'unknown';
    const opInfo = getOpInfo(op);
    const phase = item.phase || 'N/A';
    const ts = formatTimestamp(item.ts);
    const inheritedClass = isInherited ? 'inherited' : '';

    return `
        <div class="memory-call ${opInfo.type} ${inheritedClass}">
            <div class="memory-call-header">
                <div class="memory-call-info">
                    <span class="memory-call-op ${opInfo.type}">${escapeHtml(opInfo.label)}</span>
                    <span class="memory-call-phase">Phase: ${escapeHtml(phase)}</span>
                </div>
                <span class="memory-call-time">${ts}</span>
            </div>
            ${renderMemoryCallDetails(item)}
        </div>`;
}

// ===== Phase-based Organization =====

function groupByPhase(items) {
    const groups = {};
    for (const item of items) {
        const phase = item.phase || 'unknown';
        if (!groups[phase]) {
            groups[phase] = [];
        }
        groups[phase].push(item);
    }
    return groups;
}

function countByType(items) {
    const counts = { read: 0, write: 0, fork: 0, system: 0, llm: 0, total: items.length };
    for (const item of items) {
        const opInfo = getOpInfo(item.op);
        if (counts.hasOwnProperty(opInfo.type)) {
            // Exclude root creation (parent_branch_id is null) from fork count
            if (opInfo.type === 'fork') {
                const parentBranchId = item.details?.parent_branch_id;
                if (parentBranchId === null || parentBranchId === undefined) {
                    continue;  // Skip root creation, not a real fork
                }
            }
            counts[opInfo.type]++;
        }
    }
    return counts;
}

function renderPhaseSummary(phaseName, items) {
    const counts = countByType(items);
    const label = PHASE_LABELS[phaseName] || phaseName;

    return `
        <div class="phase-summary">
            <div class="phase-summary-header">
                <span class="phase-summary-title">${escapeHtml(label)}</span>
                <div class="phase-summary-stats">
                    <div class="phase-stat">
                        <div class="phase-stat-value read">${counts.read}</div>
                        <div class="phase-stat-label">Reads</div>
                    </div>
                    <div class="phase-stat">
                        <div class="phase-stat-value write">${counts.write}</div>
                        <div class="phase-stat-label">Writes</div>
                    </div>
                    <div class="phase-stat">
                        <div class="phase-stat-value total">${counts.total}</div>
                        <div class="phase-stat-label">Total</div>
                    </div>
                </div>
            </div>
        </div>`;
}

function renderPhaseGroup(phaseName, items, isExpanded = true) {
    const label = PHASE_LABELS[phaseName] || phaseName;
    const counts = countByType(items);
    const collapsedClass = isExpanded ? '' : 'collapsed';

    return `
        <div class="phase-group" data-phase="${escapeHtml(phaseName)}">
            <div class="phase-group-header" onclick="togglePhaseGroup(this)">
                <span class="phase-group-title">
                    ${escapeHtml(label)}
                    <span class="phase-group-count">${items.length} ops</span>
                </span>
                <span class="section-toggle ${collapsedClass}">▼</span>
            </div>
            <div class="phase-group-content ${collapsedClass}">
                ${items.map(item => renderMemoryCall(item)).join('')}
            </div>
        </div>`;
}

function togglePhaseGroup(header) {
    const content = header.nextElementSibling;
    const toggle = header.querySelector('.section-toggle');
    content.classList.toggle('collapsed');
    toggle.classList.toggle('collapsed');
}

// ===== Operations Summary =====

function renderOperationsSummary(memoryCalls) {
    const counts = countByType(memoryCalls);

    return `
        <div class="ops-summary">
            <div class="ops-summary-item">
                <div class="ops-summary-count read">${counts.read}</div>
                <div class="ops-summary-label">Reads</div>
            </div>
            <div class="ops-summary-item">
                <div class="ops-summary-count write">${counts.write}</div>
                <div class="ops-summary-label">Writes</div>
            </div>
            <div class="ops-summary-item">
                <div class="ops-summary-count fork">${counts.fork}</div>
                <div class="ops-summary-label">Forks</div>
            </div>
            <div class="ops-summary-item">
                <div class="ops-summary-count system">${counts.system}</div>
                <div class="ops-summary-label">System</div>
            </div>
            <div class="ops-summary-item">
                <div class="ops-summary-count llm">${counts.llm}</div>
                <div class="ops-summary-label">LLM</div>
            </div>
        </div>`;
}

// ===== Core KV Rendering =====

function renderCoreKV(items, isInherited = false) {
    if (!items || items.length === 0) return '<div class="empty-state"><div class="empty-state-text">No core memory entries</div></div>';
    const inheritedClass = isInherited ? 'inherited' : '';

    return items.map(item => `
        <div class="kv-item ${inheritedClass}">
            <span class="kv-key">${escapeHtml(item.key)}</span>
            <span class="kv-value">${escapeHtml(item.value)}</span>
        </div>`).join('');
}

// ===== Events Rendering =====

function renderEvents(items, isInherited = false) {
    if (!items || items.length === 0) return '<div class="empty-state"><div class="empty-state-text">No recall events</div></div>';
    const inheritedClass = isInherited ? 'inherited' : '';

    return items.map(item => {
        const isMemoryInjection = item.kind === 'memory_injected';
        const kindClass = isMemoryInjection ? 'memory-injection' : '';
        const taskHint = item.task_hint ? ` (${item.task_hint})` : '';
        const memorySize = item.memory_size ? ` [${item.memory_size.toLocaleString()} chars]` : '';

        return `
            <div class="event-item ${inheritedClass} ${kindClass}">
                <div class="event-kind" ${isMemoryInjection ? 'style="color: #b197fc;"' : ''}>${escapeHtml(item.kind)}${taskHint}${memorySize}</div>
                <div class="event-text">${escapeHtml(item.text)}</div>
                <div class="event-meta">
                    Phase: ${item.phase || 'N/A'} |
                    Tags: ${(item.tags || []).join(', ')}
                </div>
            </div>`;
    }).join('');
}

// ===== Archival Rendering =====

function renderArchival(items, isInherited = false) {
    if (!items || items.length === 0) return '<div class="empty-state"><div class="empty-state-text">No archival records</div></div>';
    const inheritedClass = isInherited ? 'inherited' : '';

    return items.map(item => `
        <div class="archival-item ${inheritedClass}">
            <div class="archival-tags">
                ${(item.tags || []).map(t => `<span class="archival-tag">${escapeHtml(t)}</span>`).join('')}
            </div>
            <div class="archival-text">${escapeHtml(item.text)}</div>
        </div>`).join('');
}

// ===== Ancestor Chain =====

function renderAncestorChain(ancestors, selectNodeFn) {
    if (!ancestors || ancestors.length === 0) return '';

    const links = ancestors.map(a => {
        const label = a.node_uid ? truncateText(a.node_uid, 12) : `Node ${a.index}`;
        return `<a onclick="${selectNodeFn}(${a.index})">${escapeHtml(label)}</a>`;
    });

    return `
        <div class="ancestor-chain">
            <span class="ancestor-chain-label">Inherited from:</span>
            ${links.join('<span class="ancestor-chain-arrow">→</span>')}
        </div>`;
}

// ===== Section Rendering =====

function createSection(title, content, count = 0, defaultExpanded = true) {
    const sectionId = `section-${title.toLowerCase().replace(/[^a-z0-9]/g, '-')}`;
    const collapsedClass = defaultExpanded ? '' : 'collapsed';
    const hasContent = content && !content.includes('empty-state');

    return `
        <div class="section" id="${sectionId}">
            <div class="section-header" onclick="toggleSection('${sectionId}')">
                <span class="section-title">
                    ${escapeHtml(title)}
                    ${count > 0 ? `<span class="section-badge">${count}</span>` : ''}
                </span>
                <span class="section-toggle ${collapsedClass}">▼</span>
            </div>
            <div class="section-content ${collapsedClass}">
                ${content || '<div class="empty-state"><div class="empty-state-text">No data</div></div>'}
            </div>
        </div>`;
}

function toggleSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (!section) return;

    const content = section.querySelector('.section-content');
    const toggle = section.querySelector('.section-toggle');
    content.classList.toggle('collapsed');
    toggle.classList.toggle('collapsed');
}

// Export for global use
window.MemoryDB = {
    MEMORY_OP_TYPES,
    PHASE_LABELS,
    escapeHtml,
    formatTimestamp,
    truncateText,
    getOpInfo,
    renderMemoryCall,
    renderMemoryCallDetails,
    renderFullDetails,
    toggleFullDetails,
    formatJsonValue,
    groupByPhase,
    countByType,
    renderPhaseSummary,
    renderPhaseGroup,
    togglePhaseGroup,
    renderOperationsSummary,
    renderCoreKV,
    renderEvents,
    renderArchival,
    renderAncestorChain,
    createSection,
    toggleSection,
};

// Also expose toggleFullDetails globally for onclick handlers
window.toggleFullDetails = toggleFullDetails;
