// Main JavaScript for antiNETattack

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded');
    
    // Apply dark mode immediately if enabled
    if (localStorage.getItem('darkMode') === 'enabled') {
        document.body.classList.add('dark-mode');
        console.log('Applied dark mode on page load');
    }
    
    // Global monitoring state
    let isMonitoring = false;
    
    // Check if monitoring was already active (for tab changes)
    if (sessionStorage.getItem('monitoringActive') === 'true') {
        console.log('Monitoring was active, restoring state');
        isMonitoring = true;
        
        // Re-activate monitoring on the server by checking current status first
        fetch('/api/monitoring-status')
            .then(response => response.json())
            .then(data => {
                console.log('Current monitoring status:', data);
                
                // If monitoring is not already active on the server, activate it
                if (!data.monitoring) {
                    return fetch('/api/toggle-monitoring', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({})
                    });
                } else {
                    // If already monitoring, just update UI
                    updateMonitoringUI(true);
                    return { json: () => ({ status: 'success', monitoring: true }) };
                }
            })
            .then(response => {
                if (response.json) {
                    return response.json();
                }
                return response;
            })
            .then(data => {
                console.log('Monitoring reactivated:', data);
                if (data.status === 'success') {
                    updateMonitoringUI(true);
                }
            })
            .catch(error => {
                console.error('Error reactivating monitoring:', error);
            });
    }
    
    // Monitoring toggle
    // Get monitoring UI elements
    const toggleBtn = document.getElementById('toggleMonitoring');
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    
    // Function to update UI based on monitoring state
    function updateMonitoringUI(isActive) {
        console.log('Updating monitoring UI, active:', isActive);
        
        if (toggleBtn) {
            if (isActive) {
                // Active state
                toggleBtn.classList.remove('btn-glow');
                toggleBtn.classList.add('btn-glow');
                toggleBtn.classList.add('btn-danger');
                toggleBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Monitoring';
                
                if (statusIndicator) {
                    statusIndicator.classList.remove('status-stopped');
                    statusIndicator.classList.add('status-running');
                }
                
                if (statusText) {
                    statusText.textContent = 'Running';
                }
                
                // Store monitoring state
                sessionStorage.setItem('monitoringActive', 'true');
                isMonitoring = true;
            } else {
                // Inactive state
                toggleBtn.classList.remove('btn-danger');
                toggleBtn.classList.add('btn-glow');
                toggleBtn.innerHTML = '<i class="fas fa-play"></i> Start Monitoring';
                
                if (statusIndicator) {
                    statusIndicator.classList.remove('status-running');
                    statusIndicator.classList.add('status-stopped');
                }
                
                if (statusText) {
                    statusText.textContent = 'Stopped';
                }
                
                // Store monitoring state
                sessionStorage.setItem('monitoringActive', 'false');
                isMonitoring = false;
            }
        } else {
            console.error('Toggle button not found');
        }
    }
        
    // Add event listener to monitoring toggle button if it exists
    if (toggleBtn) {
        console.log('Adding click listener to toggle button');
        toggleBtn.addEventListener('click', function(e) {
            e.preventDefault(); // Prevent default button behavior
            console.log('Toggle button clicked, current state:', isMonitoring);
            
            // Toggle monitoring state
            fetch('/api/toggle-monitoring', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.status === 'success') {
                    // Update button and status indicator
                    updateMonitoringUI(data.monitoring);
                    
                    // Store state in sessionStorage
                    sessionStorage.setItem('monitoringActive', data.monitoring);
                    isMonitoring = data.monitoring;
                    
                    // Show notification
                    const message = data.monitoring ? 
                        'Network monitoring has been started successfully.' : 
                        'Network monitoring has been stopped successfully.';
                    const title = data.monitoring ? 'Monitoring Started' : 'Monitoring Stopped';
                    showNotification(title, message);
                    
                    console.log('[INFO] Monitoring state changed:', data.monitoring ? 'Started' : 'Stopped');
                }
            })
            .catch(error => {
                console.error('[ERROR] Error toggling monitoring:', error);
                showNotification('Error', 'Failed to toggle monitoring. Please try again.', 'error');
            });
        });
    } else {
        console.error('Toggle monitoring button not found');
    }
    
    // Dark Mode Implementation
    function setupDarkMode() {
        console.log('Setting up dark mode');
        const darkModeSwitch = document.getElementById('darkModeSwitch');
        
        // Function to toggle dark mode
        function toggleDarkMode(enable) {
            console.log('Toggling dark mode:', enable);
            if (enable) {
                document.body.classList.add('dark-mode');
                localStorage.setItem('darkMode', 'enabled');
                if (darkModeSwitch) darkModeSwitch.checked = true;
                console.log('Dark mode enabled');
            } else {
                document.body.classList.remove('dark-mode');
                localStorage.setItem('darkMode', 'disabled');
                if (darkModeSwitch) darkModeSwitch.checked = false;
                console.log('Dark mode disabled');
            }
        }
        
        // Check localStorage on page load
        const darkModeEnabled = localStorage.getItem('darkMode') === 'enabled';
        toggleDarkMode(darkModeEnabled);
        
        // Set up event listener if switch exists
        if (darkModeSwitch) {
            console.log('Dark mode switch found, adding listener');
            darkModeSwitch.addEventListener('change', function() {
                toggleDarkMode(this.checked);
            });
        }
        
        // Add a global dark mode toggle function for debugging
        window.toggleDarkMode = toggleDarkMode;
        
        return { toggleDarkMode };
    }
    
    // Initialize dark mode
    const darkModeControls = setupDarkMode();
    
    // Add event listener for settings modal shown event
    const settingsModal = document.getElementById('settingsModal');
    if (settingsModal) {
        settingsModal.addEventListener('shown.bs.modal', function() {
            console.log('Settings modal shown');
            const darkModeSwitch = document.getElementById('darkModeSwitch');
            if (darkModeSwitch) {
                darkModeSwitch.checked = localStorage.getItem('darkMode') === 'enabled';
            }
        });
    }
    
    // Save settings
    const saveSettingsBtn = document.getElementById('saveSettings');
    if (saveSettingsBtn) {
        saveSettingsBtn.addEventListener('click', function() {
            // Get all settings values
            const refreshRate = document.getElementById('refreshRate').value;
            const interfaceSelect = document.getElementById('interfaceSelect').value;
            const promiscuousMode = document.getElementById('promiscuousModeSwitch').checked;
            const sensitivity = document.getElementById('sensitivityRange').value;
            const aiEnhanced = document.getElementById('aiEnhancedSwitch').checked;
            const desktopNotifications = document.getElementById('desktopNotificationsSwitch').checked;
            const emailNotifications = document.getElementById('emailNotificationsSwitch').checked;
            const emailAddress = document.getElementById('emailAddress').value;
            
            // Save settings to localStorage
            const settings = {
                refreshRate,
                interfaceSelect,
                promiscuousMode,
                sensitivity,
                aiEnhanced,
                desktopNotifications,
                emailNotifications,
                emailAddress
            };
            
            localStorage.setItem('antiNETattackSettings', JSON.stringify(settings));
            
            // Show notification
            showNotification('Settings Saved', 'Your settings have been saved successfully.');
            
            // Close modal
            const settingsModal = bootstrap.Modal.getInstance(document.getElementById('settingsModal'));
            settingsModal.hide();
        });
    }
    
    // Load settings from localStorage
    function loadSettings() {
        const savedSettings = localStorage.getItem('antiNETattackSettings');
        
        if (savedSettings) {
            const settings = JSON.parse(savedSettings);
            
            // Apply settings to form elements
            if (document.getElementById('refreshRate')) document.getElementById('refreshRate').value = settings.refreshRate || 5;
            if (document.getElementById('interfaceSelect')) document.getElementById('interfaceSelect').value = settings.interfaceSelect || 'all';
            if (document.getElementById('promiscuousModeSwitch')) document.getElementById('promiscuousModeSwitch').checked = settings.promiscuousMode !== undefined ? settings.promiscuousMode : true;
            if (document.getElementById('sensitivityRange')) document.getElementById('sensitivityRange').value = settings.sensitivity || 7;
            if (document.getElementById('aiEnhancedSwitch')) document.getElementById('aiEnhancedSwitch').checked = settings.aiEnhanced !== undefined ? settings.aiEnhanced : true;
            if (document.getElementById('desktopNotificationsSwitch')) document.getElementById('desktopNotificationsSwitch').checked = settings.desktopNotifications !== undefined ? settings.desktopNotifications : true;
            if (document.getElementById('emailNotificationsSwitch')) document.getElementById('emailNotificationsSwitch').checked = settings.emailNotifications !== undefined ? settings.emailNotifications : false;
            if (document.getElementById('emailAddress')) document.getElementById('emailAddress').value = settings.emailAddress || '';
        }
    }
    
    // Load settings when DOM is loaded
    loadSettings();
    
    // Show notification
    function showNotification(title, message, type = 'info') {
        // Check if browser supports notifications
        if (!("Notification" in window)) {
            console.log("This browser does not support desktop notifications");
            return;
        }
        
        // Check if user has granted permission
        if (Notification.permission === "granted") {
            createNotification(title, message, type);
        } else if (Notification.permission !== "denied") {
            Notification.requestPermission().then(function(permission) {
                if (permission === "granted") {
                    createNotification(title, message, type);
                }
            });
        }
    }
    
    // Create notification
    function createNotification(title, message, type) {
        const notification = new Notification(title, {
            body: message,
            icon: type === 'error' ? '/static/img/error-icon.png' : '/static/img/logo.png'
        });
        
        // Close notification after 5 seconds
        setTimeout(function() {
            notification.close();
        }, 5000);
    }
});
