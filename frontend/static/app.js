/**
 * Visual Odometry System Frontend JavaScript
 * Handles UI interactions, WebSocket communication, and 3D visualization
 */

class VisualOdometryApp {
    constructor() {
        this.ws = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.trajectoryLine = null;
        this.metricsChart = null;
        this.isProcessing = false;
        this.trajectoryData = [];

        this.init();
    }

    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.setup3DVisualization();
        this.setupMetricsChart();
        this.logMessage('System initialized', 'info');
    }

    // WebSocket Management
    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.updateConnectionStatus(true);
            this.logMessage('Connected to server', 'success');
        };

        this.ws.onclose = () => {
            this.updateConnectionStatus(false);
            this.logMessage('Disconnected from server', 'warning');
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.setupWebSocket(), 3000);
        };

        this.ws.onerror = (error) => {
            this.logMessage('WebSocket error: ' + error, 'error');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'processing_update':
                this.updateProcessingProgress(data);
                break;
            case 'error':
                this.logMessage(data.message, 'error');
                break;
            case 'status':
                this.updateSystemStatus(data);
                break;
        }
    }

    updateConnectionStatus(connected) {
        const statusIcon = document.getElementById('connection-status');
        const statusText = document.getElementById('status-text');

        if (connected) {
            statusIcon.className = 'fas fa-circle text-success';
            statusText.textContent = 'Connected';
        } else {
            statusIcon.className = 'fas fa-circle text-danger';
            statusText.textContent = 'Disconnected';
        }
    }

    // Event Listeners
    setupEventListeners() {
        // Dataset selection
        document.getElementById('dataset-select').addEventListener('change', (e) => {
            this.handleDatasetChange(e.target.value);
        });

        // Prepare sample data
        document.getElementById('prepare-sample-btn').addEventListener('click', () => {
            this.prepareSampleData();
        });

        // Start processing
        document.getElementById('process-btn').addEventListener('click', () => {
            this.startProcessing();
        });

        // Reset system
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetSystem();
        });

        // Clear console
        document.getElementById('clear-console').addEventListener('click', () => {
            this.clearConsole();
        });

        // File upload
        document.getElementById('image-upload').addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.onWindowResize();
        });
    }

    handleDatasetChange(dataset) {
        const sequenceGroup = document.getElementById('sequence-group');
        const uploadGroup = document.getElementById('upload-group');

        if (dataset === 'kitti') {
            sequenceGroup.style.display = 'block';
            uploadGroup.style.display = 'none';
        } else if (dataset === 'custom') {
            sequenceGroup.style.display = 'none';
            uploadGroup.style.display = 'block';
        } else {
            sequenceGroup.style.display = 'none';
            uploadGroup.style.display = 'none';
        }

        this.logMessage(`Dataset changed to: ${dataset}`, 'info');
    }

    // API Calls
    async prepareSampleData() {
        try {
            this.setButtonLoading('prepare-sample-btn', true);
            this.logMessage('Preparing sample data...', 'info');

            const response = await fetch('/api/prepare-sample', {
                method: 'POST'
            });

            const result = await response.json();

            if (result.success) {
                this.logMessage('Sample data prepared successfully', 'success');
            } else {
                this.logMessage('Failed to prepare sample data', 'error');
            }
        } catch (error) {
            this.logMessage('Error preparing sample data: ' + error.message, 'error');
        } finally {
            this.setButtonLoading('prepare-sample-btn', false);
        }
    }

    async startProcessing() {
        if (this.isProcessing) {
            this.logMessage('Processing already in progress', 'warning');
            return;
        }

        try {
            this.isProcessing = true;
            this.setButtonLoading('process-btn', true);
            this.logMessage('Starting visual odometry processing...', 'info');

            const config = this.getProcessingConfig();

            const response = await fetch('/api/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });

            const result = await response.json();

            if (result.success) {
                this.logMessage(`Processing completed: ${result.total_frames} frames`, 'success');
                this.updateTrajectoryVisualization(result.trajectory);
                this.updateProcessingStats(result.stats);
            } else {
                this.logMessage('Processing failed: ' + result.message, 'error');
            }
        } catch (error) {
            this.logMessage('Error during processing: ' + error.message, 'error');
        } finally {
            this.isProcessing = false;
            this.setButtonLoading('process-btn', false);
            this.updateProgressBar(0);
        }
    }

    async resetSystem() {
        try {
            this.logMessage('Resetting system...', 'info');

            const response = await fetch('/api/reset', {
                method: 'POST'
            });

            const result = await response.json();

            if (result.success) {
                this.clearTrajectory();
                this.clearStats();
                this.updateProgressBar(0);
                this.logMessage('System reset successfully', 'success');
            }
        } catch (error) {
            this.logMessage('Error resetting system: ' + error.message, 'error');
        }
    }

    async handleFileUpload(files) {
        if (files.length === 0) return;

        try {
            this.logMessage(`Uploading ${files.length} images...`, 'info');

            const formData = new FormData();
            for (let file of files) {
                formData.append('files', file);
            }

            const response = await fetch('/api/upload-images', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.logMessage(result.message, 'success');
                // Switch to custom dataset
                document.getElementById('dataset-select').value = 'custom';
            } else {
                this.logMessage('Upload failed: ' + result.message, 'error');
            }
        } catch (error) {
            this.logMessage('Error uploading files: ' + error.message, 'error');
        }
    }

    getProcessingConfig() {
        return {
            camera: {
                fx: parseFloat(document.getElementById('fx').value),
                fy: parseFloat(document.getElementById('fy').value),
                cx: parseFloat(document.getElementById('cx').value),
                cy: parseFloat(document.getElementById('cy').value),
                baseline: 0.075
            },
            mode: document.getElementById('mode-select').value,
            detector: document.getElementById('detector-select').value,
            dataset: document.getElementById('dataset-select').value,
            sequence: document.getElementById('sequence-select').value
        };
    }

    // 3D Visualization
    setup3DVisualization() {
        const container = document.getElementById('trajectory-container');
        const width = container.clientWidth;
        const height = container.clientHeight;

        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);

        // Camera setup
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        this.camera.position.set(10, 10, 10);
        this.camera.lookAt(0, 0, 0);

        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(this.renderer.domElement);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);

        // Grid
        const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x444444);
        this.scene.add(gridHelper);

        // Axes helper
        const axesHelper = new THREE.AxesHelper(5);
        this.scene.add(axesHelper);

        // Controls (basic mouse interaction)
        this.setupCameraControls();

        // Start render loop
        this.animate();
    }

    setupCameraControls() {
        let mouseDown = false;
        let mouseX = 0;
        let mouseY = 0;

        this.renderer.domElement.addEventListener('mousedown', (event) => {
            mouseDown = true;
            mouseX = event.clientX;
            mouseY = event.clientY;
        });

        this.renderer.domElement.addEventListener('mouseup', () => {
            mouseDown = false;
        });

        this.renderer.domElement.addEventListener('mousemove', (event) => {
            if (!mouseDown) return;

            const deltaX = event.clientX - mouseX;
            const deltaY = event.clientY - mouseY;

            // Rotate camera around origin
            const spherical = new THREE.Spherical();
            spherical.setFromVector3(this.camera.position);
            spherical.theta -= deltaX * 0.01;
            spherical.phi += deltaY * 0.01;
            spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));

            this.camera.position.setFromSpherical(spherical);
            this.camera.lookAt(0, 0, 0);

            mouseX = event.clientX;
            mouseY = event.clientY;
        });

        // Zoom with mouse wheel
        this.renderer.domElement.addEventListener('wheel', (event) => {
            const scale = event.deltaY > 0 ? 1.1 : 0.9;
            this.camera.position.multiplyScalar(scale);
        });
    }

    updateTrajectoryVisualization(trajectory) {
        // Remove existing trajectory
        if (this.trajectoryLine) {
            this.scene.remove(this.trajectoryLine);
        }

        if (trajectory.length === 0) return;

        // Create trajectory line
        const points = [];
        for (let point of trajectory) {
            points.push(new THREE.Vector3(
                point.position[0],
                point.position[1],
                point.position[2]
            ));
        }

        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: 0x00ff00,
            linewidth: 3
        });

        this.trajectoryLine = new THREE.Line(geometry, material);
        this.scene.add(this.trajectoryLine);

        // Add start and end markers
        if (points.length > 0) {
            // Start marker (green sphere)
            const startGeometry = new THREE.SphereGeometry(0.2, 8, 6);
            const startMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
            const startMarker = new THREE.Mesh(startGeometry, startMaterial);
            startMarker.position.copy(points[0]);
            this.scene.add(startMarker);

            // End marker (red sphere)
            const endGeometry = new THREE.SphereGeometry(0.2, 8, 6);
            const endMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
            const endMarker = new THREE.Mesh(endGeometry, endMaterial);
            endMarker.position.copy(points[points.length - 1]);
            this.scene.add(endMarker);
        }

        this.trajectoryData = trajectory;
        this.logMessage(`Trajectory updated: ${trajectory.length} poses`, 'info');
    }

    clearTrajectory() {
        if (this.trajectoryLine) {
            this.scene.remove(this.trajectoryLine);
            this.trajectoryLine = null;
        }

        // Remove markers
        const objectsToRemove = [];
        this.scene.traverse((child) => {
            if (child instanceof THREE.Mesh && child.geometry instanceof THREE.SphereGeometry) {
                objectsToRemove.push(child);
            }
        });

        objectsToRemove.forEach(obj => this.scene.remove(obj));
        this.trajectoryData = [];
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        const container = document.getElementById('trajectory-container');
        const width = container.clientWidth;
        const height = container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    // Metrics Chart
    setupMetricsChart() {
        const ctx = document.getElementById('metrics-chart').getContext('2d');

        this.metricsChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Keypoints',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }, {
                    label: 'Matches',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                    }
                }
            }
        });
    }

    updateProcessingStats(stats) {
        if (!stats || stats.length === 0) return;

        // Update real-time stats display
        const latestStats = stats[stats.length - 1];
        document.getElementById('keypoints-count').textContent = latestStats.keypoints_count || 0;
        document.getElementById('matches-count').textContent = latestStats.matches_count || 0;
        document.getElementById('frames-processed').textContent = stats.length;

        // Update metrics chart
        const labels = stats.map((_, i) => i.toString());
        const keypointsData = stats.map(s => s.keypoints_count || 0);
        const matchesData = stats.map(s => s.matches_count || 0);

        this.metricsChart.data.labels = labels;
        this.metricsChart.data.datasets[0].data = keypointsData;
        this.metricsChart.data.datasets[1].data = matchesData;
        this.metricsChart.update();
    }

    updateProcessingProgress(data) {
        this.updateProgressBar(data.progress);

        // Update real-time stats
        document.getElementById('frames-processed').textContent = data.frame;
        document.getElementById('keypoints-count').textContent = data.stats.keypoints_count || 0;
        document.getElementById('matches-count').textContent = data.stats.matches_count || 0;

        // Add trajectory point if available
        if (data.trajectory_point) {
            // Add point to visualization in real-time
            this.addTrajectoryPoint(data.trajectory_point);
        }

        this.logMessage(`Processing frame ${data.frame}/${data.total_frames}`, 'info');
    }

    addTrajectoryPoint(point) {
        // This would add a single point to the trajectory in real-time
        // Implementation depends on specific requirements
    }

    // UI Utilities
    updateProgressBar(progress) {
        const progressBar = document.getElementById('progress-bar');
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
    }

    setButtonLoading(buttonId, loading) {
        const button = document.getElementById(buttonId);
        const originalText = button.innerHTML;

        if (loading) {
            button.innerHTML = '<span class="loading-spinner"></span> Processing...';
            button.disabled = true;
        } else {
            button.innerHTML = originalText;
            button.disabled = false;
        }
    }

    clearStats() {
        document.getElementById('keypoints-count').textContent = '0';
        document.getElementById('matches-count').textContent = '0';
        document.getElementById('frames-processed').textContent = '0';

        if (this.metricsChart) {
            this.metricsChart.data.labels = [];
            this.metricsChart.data.datasets[0].data = [];
            this.metricsChart.data.datasets[1].data = [];
            this.metricsChart.update();
        }
    }

    updateSystemStatus(status) {
        // Update system status display
        this.logMessage(`System status: Pipeline ${status.pipeline_active ? 'active' : 'inactive'}, ${status.connected_clients} clients`, 'info');
    }

    // Console Logging
    logMessage(message, type = 'info') {
        const console = document.getElementById('console-output');
        const timestamp = new Date().toLocaleTimeString();

        const line = document.createElement('div');
        line.className = `console-line ${type}`;
        line.innerHTML = `<span class="timestamp">[${timestamp}]</span>${message}`;

        console.appendChild(line);
        console.scrollTop = console.scrollHeight;

        // Keep only last 100 messages
        while (console.children.length > 100) {
            console.removeChild(console.firstChild);
        }
    }

    clearConsole() {
        document.getElementById('console-output').innerHTML = '';
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.voApp = new VisualOdometryApp();
});