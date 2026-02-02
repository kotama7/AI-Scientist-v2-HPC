/**
 * Tree Canvas Component using p5.js
 * Renders the memory tree visualization with interactive nodes
 */

function createTreeSketch(memoryData, callbacks = {}) {
    const {
        onNodeSelect = () => {},
        onNodeHover = () => {},
    } = callbacks;

    return function(p) {
        let selectedNodeIndex = -1;
        let hoveredNodeIndex = -1;
        let canvasWidth, canvasHeight;
        let panX = 0, panY = 0;
        let zoom = 1;
        let isDragging = false;
        let lastMouseX, lastMouseY;

        const NODE_RADIUS = 12;
        const SELECTED_RADIUS = 16;

        // Colors
        const COLORS = {
            background: '#16213e',
            edge: '#3a3f5c',
            node: '#4dabf7',
            nodeHover: '#74c0fc',
            nodeSelected: '#e94560',
            nodeBest: '#ffd700',
            nodeBestGlow: 'rgba(255, 215, 0, 0.3)',
            nodeVirtual: '#555',
            text: '#eee',
            textMuted: '#888',
        };

        p.setup = function() {
            const container = document.getElementById('canvas-container');
            canvasWidth = container.offsetWidth;
            canvasHeight = container.offsetHeight;
            p.createCanvas(canvasWidth, canvasHeight);
            p.textFont('system-ui, -apple-system, sans-serif');
        };

        p.draw = function() {
            p.background(COLORS.background);
            p.push();
            p.translate(panX, panY);
            p.scale(zoom);

            // Draw edges
            drawEdges();

            // Draw nodes
            drawNodes();

            p.pop();

            // Draw UI overlay
            drawOverlay();
        };

        function drawEdges() {
            p.stroke(COLORS.edge);
            p.strokeWeight(1.5 / zoom);
            p.noFill();

            const { edges, layout } = memoryData;
            if (!edges || !layout) return;

            for (const [from, to] of edges) {
                if (from >= layout.length || to >= layout.length) continue;

                const x1 = layout[from][0] * canvasWidth;
                const y1 = layout[from][1] * canvasHeight;
                const x2 = layout[to][0] * canvasWidth;
                const y2 = layout[to][1] * canvasHeight;

                // Draw curved line
                p.beginShape();
                p.vertex(x1, y1);
                p.bezierVertex(x1, (y1 + y2) / 2, x2, (y1 + y2) / 2, x2, y2);
                p.endShape();
            }
        }

        function drawNodes() {
            const { nodes, layout } = memoryData;
            if (!nodes || !layout) return;

            for (let i = 0; i < nodes.length; i++) {
                const node = nodes[i];
                if (i >= layout.length) continue;

                const x = layout[i][0] * canvasWidth;
                const y = layout[i][1] * canvasHeight;
                const isSelected = i === selectedNodeIndex;
                const isHovered = i === hoveredNodeIndex;
                const isVirtual = node.is_virtual;
                const isBest = node.is_best;

                // Determine node color and size
                let nodeColor, radius;
                if (isSelected) {
                    nodeColor = COLORS.nodeSelected;
                    radius = SELECTED_RADIUS;
                } else if (isBest) {
                    nodeColor = COLORS.nodeBest;
                    radius = NODE_RADIUS + 3;
                } else if (isHovered) {
                    nodeColor = COLORS.nodeHover;
                    radius = NODE_RADIUS + 2;
                } else if (isVirtual) {
                    nodeColor = COLORS.nodeVirtual;
                    radius = NODE_RADIUS - 2;
                } else {
                    nodeColor = COLORS.node;
                    radius = NODE_RADIUS;
                }

                // Draw glow for best nodes
                if (isBest && !isSelected) {
                    p.noStroke();
                    p.fill(COLORS.nodeBestGlow);
                    p.ellipse(x, y, (radius + 8) * 2 / zoom);
                }

                // Draw node
                p.noStroke();
                p.fill(nodeColor);
                p.ellipse(x, y, radius * 2 / zoom);

                // Draw star marker for best nodes
                if (isBest) {
                    p.fill(COLORS.background);
                    p.noStroke();
                    p.textSize(10 / zoom);
                    p.textAlign(p.CENTER, p.CENTER);
                    p.text('\u2605', x, y - 1 / zoom); // ★
                }

                // Draw selection ring
                if (isSelected) {
                    p.noFill();
                    p.stroke(COLORS.nodeSelected);
                    p.strokeWeight(2 / zoom);
                    p.ellipse(x, y, (radius + 4) * 2 / zoom);
                } else if (isBest) {
                    p.noFill();
                    p.stroke(COLORS.nodeBest);
                    p.strokeWeight(2 / zoom);
                    p.ellipse(x, y, (radius + 4) * 2 / zoom);
                }

                // Draw label for selected/hovered/best node
                if (isSelected || isHovered || isBest) {
                    drawNodeLabel(node, x, y, isBest);
                }
            }
        }

        function drawNodeLabel(node, x, y, isBest) {
            const label = node.node_uid ?
                node.node_uid.substring(0, 8) + '...' :
                `Node ${node.index}`;

            const labelY = y - NODE_RADIUS / zoom - 5 / zoom;

            if (isBest) {
                // Draw "BEST" badge above the node label
                const badgeText = node.best_stages && node.best_stages.length > 0
                    ? '\u2605 BEST'
                    : '\u2605 BEST';
                p.fill(COLORS.nodeBest);
                p.noStroke();
                p.textSize(9 / zoom);
                p.textAlign(p.CENTER, p.BOTTOM);
                p.text(badgeText, x, labelY - 12 / zoom);
            }

            p.fill(COLORS.text);
            p.noStroke();
            p.textSize(11 / zoom);
            p.textAlign(p.CENTER, p.BOTTOM);
            p.text(label, x, labelY);
        }

        function drawOverlay() {
            // Draw zoom level indicator
            p.fill(COLORS.textMuted);
            p.noStroke();
            p.textSize(10);
            p.textAlign(p.LEFT, p.BOTTOM);
            p.text(`Zoom: ${(zoom * 100).toFixed(0)}%`, 10, canvasHeight - 10);

            // Draw instructions
            p.textAlign(p.RIGHT, p.BOTTOM);
            p.text('Scroll to zoom, drag to pan, double-click to reset', canvasWidth - 10, canvasHeight - 10);
        }

        function getNodeAtPosition(mx, my) {
            const { nodes, layout } = memoryData;
            if (!nodes || !layout) return -1;

            // Transform mouse coordinates
            const worldX = (mx - panX) / zoom;
            const worldY = (my - panY) / zoom;

            for (let i = nodes.length - 1; i >= 0; i--) {
                if (i >= layout.length) continue;

                const x = layout[i][0] * canvasWidth;
                const y = layout[i][1] * canvasHeight;
                const dist = p.dist(worldX, worldY, x, y);

                if (dist < NODE_RADIUS * 1.5) {
                    return i;
                }
            }
            return -1;
        }

        p.mousePressed = function() {
            if (p.mouseX < 0 || p.mouseX > canvasWidth ||
                p.mouseY < 0 || p.mouseY > canvasHeight) return;

            const nodeIndex = getNodeAtPosition(p.mouseX, p.mouseY);
            if (nodeIndex >= 0) {
                selectedNodeIndex = nodeIndex;
                onNodeSelect(nodeIndex, memoryData.nodes[nodeIndex]);
            } else {
                isDragging = true;
                lastMouseX = p.mouseX;
                lastMouseY = p.mouseY;
            }
        };

        p.mouseReleased = function() {
            isDragging = false;
        };

        p.mouseDragged = function() {
            if (isDragging) {
                panX += p.mouseX - lastMouseX;
                panY += p.mouseY - lastMouseY;
                lastMouseX = p.mouseX;
                lastMouseY = p.mouseY;
            }
        };

        p.mouseMoved = function() {
            const nodeIndex = getNodeAtPosition(p.mouseX, p.mouseY);
            if (nodeIndex !== hoveredNodeIndex) {
                hoveredNodeIndex = nodeIndex;
                p.cursor(nodeIndex >= 0 ? 'pointer' : 'default');
                if (nodeIndex >= 0) {
                    onNodeHover(nodeIndex, memoryData.nodes[nodeIndex]);
                }
            }
        };

        p.mouseWheel = function(event) {
            if (p.mouseX < 0 || p.mouseX > canvasWidth ||
                p.mouseY < 0 || p.mouseY > canvasHeight) return;

            const zoomSensitivity = 0.001;
            const zoomDelta = -event.delta * zoomSensitivity;
            const newZoom = p.constrain(zoom * (1 + zoomDelta), 0.3, 3);

            // Zoom towards mouse position
            const mouseXWorld = (p.mouseX - panX) / zoom;
            const mouseYWorld = (p.mouseY - panY) / zoom;

            zoom = newZoom;

            panX = p.mouseX - mouseXWorld * zoom;
            panY = p.mouseY - mouseYWorld * zoom;

            return false; // Prevent page scroll
        };

        p.doubleClicked = function() {
            // Reset view
            panX = 0;
            panY = 0;
            zoom = 1;
        };

        p.windowResized = function() {
            const container = document.getElementById('canvas-container');
            canvasWidth = container.offsetWidth;
            canvasHeight = container.offsetHeight;
            p.resizeCanvas(canvasWidth, canvasHeight);
        };

        // Public method to select a node programmatically
        p.selectNode = function(index) {
            if (index >= 0 && index < memoryData.nodes.length) {
                selectedNodeIndex = index;
                onNodeSelect(index, memoryData.nodes[index]);
            }
        };

        return p;
    };
}

// Export
window.createTreeSketch = createTreeSketch;
