/**
 * Resizable Panel Component
 * Allows users to drag the divider between two panels to resize them.
 */

class ResizablePanel {
    constructor(options = {}) {
        this.leftPanel = options.leftPanel || document.querySelector('.panel-left');
        this.rightPanel = options.rightPanel || document.querySelector('.panel-right');
        this.resizer = options.resizer || document.querySelector('.panel-resizer');
        this.minLeftWidth = options.minLeftWidth || 200;
        this.maxLeftWidth = options.maxLeftWidth || window.innerWidth * 0.7;
        this.storageKey = options.storageKey || 'panel-width';

        this.isResizing = false;
        this.startX = 0;
        this.startWidth = 0;

        this.init();
    }

    init() {
        if (!this.resizer || !this.leftPanel) {
            console.warn('ResizablePanel: Required elements not found');
            return;
        }

        // Restore saved width
        const savedWidth = localStorage.getItem(this.storageKey);
        if (savedWidth) {
            this.setLeftWidth(parseInt(savedWidth, 10));
        }

        // Mouse events
        this.resizer.addEventListener('mousedown', this.startResize.bind(this));
        document.addEventListener('mousemove', this.doResize.bind(this));
        document.addEventListener('mouseup', this.stopResize.bind(this));

        // Touch events for mobile
        this.resizer.addEventListener('touchstart', this.startResizeTouch.bind(this));
        document.addEventListener('touchmove', this.doResizeTouch.bind(this));
        document.addEventListener('touchend', this.stopResize.bind(this));

        // Double-click to reset
        this.resizer.addEventListener('dblclick', this.resetWidth.bind(this));

        // Handle window resize
        window.addEventListener('resize', this.handleWindowResize.bind(this));
    }

    startResize(e) {
        this.isResizing = true;
        this.startX = e.clientX;
        this.startWidth = this.leftPanel.offsetWidth;
        this.resizer.classList.add('active');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    }

    startResizeTouch(e) {
        if (e.touches.length === 1) {
            this.isResizing = true;
            this.startX = e.touches[0].clientX;
            this.startWidth = this.leftPanel.offsetWidth;
            this.resizer.classList.add('active');
            e.preventDefault();
        }
    }

    doResize(e) {
        if (!this.isResizing) return;

        const dx = e.clientX - this.startX;
        const newWidth = this.startWidth + dx;
        this.setLeftWidth(newWidth);
    }

    doResizeTouch(e) {
        if (!this.isResizing || e.touches.length !== 1) return;

        const dx = e.touches[0].clientX - this.startX;
        const newWidth = this.startWidth + dx;
        this.setLeftWidth(newWidth);
        e.preventDefault();
    }

    stopResize() {
        if (!this.isResizing) return;

        this.isResizing = false;
        this.resizer.classList.remove('active');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';

        // Save width to localStorage
        localStorage.setItem(this.storageKey, this.leftPanel.offsetWidth);

        // Trigger resize event for p5.js canvas
        window.dispatchEvent(new Event('resize'));
    }

    setLeftWidth(width) {
        const clampedWidth = Math.max(this.minLeftWidth, Math.min(this.maxLeftWidth, width));
        this.leftPanel.style.width = `${clampedWidth}px`;
        this.leftPanel.style.flexBasis = `${clampedWidth}px`;
    }

    resetWidth() {
        const defaultWidth = window.innerWidth * 0.35;
        this.setLeftWidth(defaultWidth);
        localStorage.removeItem(this.storageKey);
        window.dispatchEvent(new Event('resize'));
    }

    handleWindowResize() {
        // Update max width on window resize
        this.maxLeftWidth = window.innerWidth * 0.7;

        // Clamp current width if needed
        const currentWidth = this.leftPanel.offsetWidth;
        if (currentWidth > this.maxLeftWidth) {
            this.setLeftWidth(this.maxLeftWidth);
        }
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ResizablePanel;
}
