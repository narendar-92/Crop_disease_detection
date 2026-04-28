// Modern JavaScript for Crop Disease Detection UI

class CropDiseaseDetector {
    constructor() {
        this.selectedFiles = [];
        this.currentTheme = localStorage.getItem('theme') || 'light';
        this.init();
    }

    init() {
        this.bindElements();
        this.setupEventListeners();
        this.applyTheme();
        this.checkExistingResults();
        this.updateAnalyzeButton();
    }

    bindElements() {
        this.uploadArea = document.getElementById('upload-area');
        this.fileInput = document.getElementById('file-input');
        this.previewSection = document.getElementById('preview-section');
        this.imageGrid = document.getElementById('image-grid');
        this.analyzeBtn = document.getElementById('analyze-btn');
        this.resetBtn = document.getElementById('reset-btn');
        this.resultsSection = document.getElementById('results-section');
        this.themeToggle = document.getElementById('theme-toggle');
        this.cropSelect = document.getElementById('crop-select');
        this.predictionForm = document.getElementById('prediction-form');
        this.formCrop = document.getElementById('form-crop');
        this.formImage = document.getElementById('form-image');
    }

    setupEventListeners() {
        // Upload area events
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));

        // File input change
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));

        // Browse link
        document.querySelector('.browse-link').addEventListener('click', (e) => {
            e.stopPropagation();
            this.fileInput.click();
        });

        // Buttons
        this.analyzeBtn.addEventListener('click', this.analyzeDisease.bind(this));
        this.resetBtn.addEventListener('click', this.reset.bind(this));

        // Theme toggle
        this.themeToggle.addEventListener('click', this.toggleTheme.bind(this));

        // Crop selection
        this.cropSelect.addEventListener('change', this.updateAnalyzeButton.bind(this));
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');

        const files = Array.from(e.dataTransfer.files);
        this.processFiles(files);
    }

    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        this.processFiles(files);
    }

    processFiles(files) {
        const validFiles = files.filter(file => {
            const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
            return validTypes.includes(file.type);
        });

        if (validFiles.length === 0) {
            this.showNotification('Please select a valid image file (PNG, JPG, JPEG, WEBP)', 'error');
            return;
        }

        // Only allow single file for backend compatibility
        this.selectedFiles = [validFiles[0]];
        this.updatePreview();
        this.updateAnalyzeButton();
    }

    updatePreview() {
        if (this.selectedFiles.length === 0) {
            this.previewSection.style.display = 'none';
            return;
        }

        this.previewSection.style.display = 'block';
        this.imageGrid.innerHTML = '';

        this.selectedFiles.forEach((file, index) => {
            const previewDiv = document.createElement('div');
            previewDiv.className = 'image-preview';

            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            img.alt = file.name;

            const infoDiv = document.createElement('div');
            infoDiv.className = 'image-info';

            const nameSpan = document.createElement('div');
            nameSpan.className = 'image-name';
            nameSpan.textContent = file.name;

            const removeBtn = document.createElement('button');
            removeBtn.className = 'remove-btn';
            removeBtn.innerHTML = '×';
            removeBtn.addEventListener('click', () => this.removeFile(index));

            infoDiv.appendChild(nameSpan);
            previewDiv.appendChild(img);
            previewDiv.appendChild(infoDiv);
            previewDiv.appendChild(removeBtn);

            this.imageGrid.appendChild(previewDiv);
        });
    }

    removeFile(index) {
        this.selectedFiles.splice(index, 1);
        this.updatePreview();
        this.updateAnalyzeButton();
    }

    updateAnalyzeButton() {
        const hasFiles = this.selectedFiles.length > 0;
        const hasCrop = this.cropSelect.value !== '';

        this.analyzeBtn.disabled = !hasFiles || !hasCrop;

        if (hasFiles && hasCrop) {
            this.analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Disease';
        }
    }

    async analyzeDisease() {
        if (this.selectedFiles.length === 0 || !this.cropSelect.value) return;

        // Show loading state
        this.setLoadingState(true);
        this.showResultsSection();

        try {
            // Prepare the hidden form with selected data
            this.formCrop.value = this.cropSelect.value;

            // Create a new DataTransfer to set the file input
            const dt = new DataTransfer();
            dt.items.add(this.selectedFiles[0]);
            this.formImage.files = dt.files;

            // Submit the form normally (this will work with Flask backend)
            this.predictionForm.submit();

        } catch (error) {
            console.error('Error:', error);
            this.showError('Analysis failed. Please try again.');
            this.setLoadingState(false);
        }
    }

    setLoadingState(loading) {
        if (loading) {
            this.analyzeBtn.disabled = true;
            this.analyzeBtn.innerHTML = '<div class="spinner"></div> Analyzing...';
            this.resetBtn.style.display = 'none';
        } else {
            this.updateAnalyzeButton();
            this.resetBtn.style.display = 'inline-flex';
        }
    }

    showResultsSection() {
        // Results will be shown by Flask template rendering
    }

    showError(message) {
        // Error handling is done by Flask backend with warning messages
        console.error(message);
    }

    reset() {
        this.selectedFiles = [];
        this.fileInput.value = '';
        this.updatePreview();
        this.updateAnalyzeButton();
        this.resultsSection.style.display = 'none';
        this.resetBtn.style.display = 'none';

        // Reset status
        const statusBadge = document.getElementById('status-badge');
        statusBadge.textContent = 'Analyzing...';
        statusBadge.className = 'status-badge';
    }

    toggleTheme() {
        this.currentTheme = this.currentTheme === 'light' ? 'dark' : 'light';
        this.applyTheme();
        localStorage.setItem('theme', this.currentTheme);
    }

    applyTheme() {
        document.documentElement.setAttribute('data-theme', this.currentTheme);
        const icon = this.themeToggle.querySelector('i');

        if (this.currentTheme === 'dark') {
            icon.className = 'fas fa-sun';
        } else {
            icon.className = 'fas fa-moon';
        }
    }

    showNotification(message, type = 'info') {
        // Simple notification - you could enhance this with a proper notification system
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'error' ? '#d32f2f' : '#2e7d32'};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
            z-index: 1001;
            animation: slideIn 0.3s ease;
        `;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// Add notification animations to CSS dynamically
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CropDiseaseDetector();
});

// Add some utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Add smooth scrolling for better UX
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});