/**
 * Start capturing and sending video frames
 */


const sidebarContainer = document.getElementById('sidebarContainer');
const sidebarToggle = document.getElementById('sidebarToggle');
const toggleIcon = document.getElementById('toggleIcon');
const historyPanel = document.getElementById('historyPanel');
const historyToggle = document.getElementById('historyToggle');
const historyToggleIcon = document.getElementById('historyToggleIcon');
    
    // Gesture history array
    let gestureHistory = [];

const gestureIcons = {
    "Doing other things": "fa-ellipsis-h", // Placeholder
    "No gesture": "fa-ellipsis-h",
    "Rolling Hand Backward": "fa-caret-down",
    "Rolling Hand Forward": "fa-caret-up",
    "Shaking Hand": "fa-arrows-up-down",
    "Sliding Two Fingers Down": "fa-arrow-down", // Closest icon
    "Sliding Two Fingers Left": "fa-arrow-left", // Closest icon
    "Sliding Two Fingers Right": "fa-arrow-right", // Closest icon
    "Sliding Two Fingers Up": "fa-arrow-up", // Closest icon
    "Stop Sign": "fa-ban",
    "Swiping Down": "fa-arrow-down",
    "Swiping Left": "fa-arrow-left",
    "Swiping Right": "fa-arrow-right",
    "Swiping Up": "fa-arrow-up",
    "Thumb Down": "fa-thumbs-down",
    "Thumb Up": "fa-thumbs-up",
    "Turning Hand Clockwise": "fa-rotate-right",
    "Turning Hand Counterclockwise": "fa-rotate-left"
};


function startCapturingFrames() {
  if (isCapturing) return;
  
  isCapturing = true;
  console.log('Starting frame capture');
  
  // Calculate the interval based on the desired frame rate
  const frameDelay = 1000 / FRAME_RATE;
  
  frameInterval = setInterval(() => {
    if (!isConnected || !isCapturing) return;
    
    // Draw the current video frame onto the canvas
    captureContext.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);
    
    // Get the frame data as a blob
    captureCanvas.toBlob((blob) => {
        if (blob && socket && socket.readyState === WebSocket.OPEN) {
            // Convert Blob to base64
            const reader = new FileReader();
            reader.onloadend = () => {
                // reader.result is a base64 string (e.g., "data:image/jpeg;base64,...")
                // Remove the data URI prefix to send only the base64 data
                const base64String = reader.result.split(',')[1];
                socket.send(base64String);
            };
            reader.readAsDataURL(blob); // Convert Blob to base64
        }
    }, 'image/jpeg', 0.7); // Use JPEG with 70% quality for better compression

}, frameDelay);
}

/**
 * Stop capturing video frames
 */
function stopCapturingFrames() {
  if (!isCapturing) return;
  
  console.log('Stopping frame capture');
  clearInterval(frameInterval);
  isCapturing = false;
  
  // Clean up video resources
  if (videoElement) {
    videoElement.pause();
    videoElement.srcObject = null;
  }
  
  if (videoStream) {
    videoStream.getTracks().forEach(track => track.stop());
    videoStream = null;
  }
  
  // Remove video element from the DOM
  if (videoElement && videoElement.parentNode) {
    videoElement.parentNode.removeChild(videoElement);
  }
  
  // Reset the placeholder content
  videoPlaceholder.innerHTML = `
    <div class="placeholder-content">
      <i class="fas fa-video placeholder-icon"></i>
      <p>Camera feed will appear here</p>
    </div>
  `;
}// MotionSpeak WebSocket Integration
// This script manages the WebSocket connection for the MotionSpeak gesture detection application
// and handles frame capture and transmission

// Extract parameters from the URL if needed
const urlParams = new URLSearchParams(window.location.search);
const serverParam = urlParams.get('server') || 'wss://166.113.52.39:47482';
const sessionId = urlParams.get('session') || generateSessionId();

// WebSocket connection
let socket;
let isConnected = false;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY = 3000; // 3 seconds

// Frame capture variables
let videoStream;
let videoElement;
let captureCanvas;
let captureContext;
let frameInterval;
const FRAME_RATE = 15; // Frames per second to send
let isCapturing = false;

// DOM Elements references - using the IDs from po.html
const startButton = document.getElementById('startButton');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const legendBox = document.getElementById('legendBox');
const emojiDisplay = document.getElementById('emojiDisplay');
const historyList = document.getElementById('historyList');

// Initialize connection when the Start Detection button is clicked
if (startButton) {
  startButton.addEventListener('click', function() {
    if (videoPlaceholder.classList.contains('video-active')) {
      // User clicked Stop Detection
      stopCapturingFrames();
      closeWebSocketConnection();
    } else {
      // User clicked Start Detection
      initializeWebSocketConnection();
    }
  });
}

/**
 * Initialize WebSocket connection
 */
function initializeWebSocketConnection() {
  // Only attempt to connect if we're not already connected
  if (!isConnected) {
    try {
      // Create WebSocket connection
      socket = new WebSocket(`${serverParam}/ws/stream/`);
      
      // WebSocket event handlers
      socket.onopen = handleSocketOpen;
      socket.onmessage = handleSocketMessage;
      socket.onclose = handleSocketClose;
      socket.onerror = handleSocketError;
      
      console.log('WebSocket connection initializing...');
    } catch (error) {
      console.error('Failed to initialize WebSocket connection:', error);
      handleConnectionFailure('Failed to connect to gesture recognition server.');
    }
  }
}

/**
 * Initialize video capture from webcam
 */
// Function to display a gesture in the emoji display
function displayGesture(gestureData) {
    // Check if we received an object with prediction or just a string with the gesture name
    const gestureName = typeof gestureData === 'object' ? gestureData.prediction : gestureData;
    const confidence = typeof gestureData === 'object' ? gestureData.confidence : null;
    
    const iconClass = gestureIcons[gestureName];
    
    if (iconClass) {
        // Create gesture display with confidence if available
        let displayHtml = `
            <div class="emoji-icon">
                <i class="fas ${iconClass}"></i>
            </div>
            <div class="emoji-label">${gestureName}</div>
        `;
        
        // Add confidence information if available
        if (confidence !== null) {
            // Format confidence as percentage with 1 decimal place
            const confidencePercent = (confidence * 100).toFixed(1);
            displayHtml += `<div class="emoji-confidence">${confidencePercent}% confidence</div>`;
        }
        
        emojiDisplay.innerHTML = displayHtml;
        emojiDisplay.classList.add('active');
        
        // Add to history if detection is active
        if (videoPlaceholder.classList.contains('video-active')) {
            addGestureToHistory(gestureData);
        }
    } else {
        emojiDisplay.innerHTML = '<div class="emoji-placeholder">Gesture emoji</div>';
        emojiDisplay.classList.remove('active');
    }
}

 function addGestureToHistory(gestureName) {
      const timestamp = new Date();
      const historyItem = {
        gesture: gestureName,
        timestamp: timestamp,
        icon: gestureIcons[gestureName] || 'fa-question'
      };
      
      // Add to our history array
      gestureHistory.unshift(historyItem);
      
      // Update the history list in the UI
      updateHistoryList();
    }
/**
 * Initialize video capture from webcam
 */
async function initializeVideoCapture() {
    try {
        // Request user permission to access the camera
        let videoDevices = [];
        const devices = await navigator.mediaDevices.enumerateDevices();
        for (const device of devices) {
            if (device.kind === 'videoinput') {
                console.log('Camera found:', device.kind);
                videoDevices.push(device);
            }
        }
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: { 
                facingMode: "user", // Use the front camera if available
            },
        });
        
        console.log('Camera access granted');
        
        // Create video element to display the stream
        videoElement = document.createElement('video');
        videoElement.srcObject = videoStream;
        videoElement.autoplay = true;
        videoElement.playsInline = true;
        videoElement.muted = true;
        videoElement.style.width = '100%';
        videoElement.style.height = '100%';
        videoElement.style.objectFit = 'cover';
        
        // Create canvas for frame capture
        captureCanvas = document.createElement('canvas');
        captureCanvas.width = 640;
        captureCanvas.height = 480;
        captureContext = captureCanvas.getContext('2d');
        
        // Clear the placeholder and append the video
        videoPlaceholder.innerHTML = ''; // Clear any existing content
        videoPlaceholder.appendChild(videoElement);
        
        // Start capturing frames once video is ready
        videoElement.onloadeddata = () => {
            startCapturingFrames();
        };
        
        return true;
    } catch (error) {
        console.error('Failed to access camera:', error);
        handleConnectionFailure('Unable to access camera. Please make sure camera permissions are granted.');
        return false;
    }
}

// Updated function to update the history list in the UI to include new data
function updateHistoryList() {
    if (gestureHistory.length === 0) {
        historyList.innerHTML = '<li class="history-empty">No gestures detected yet.</li>';
        return;
    }
    
    historyList.innerHTML = '';
    
    gestureHistory.forEach((item, index) => {
        const li = document.createElement('li');
        li.className = 'history-item';
        if (index === 0) li.classList.add('highlight');
        
        // Create base history item HTML
        let historyHtml = `
            <div class="history-icon">
                <i class="fas ${item.icon}"></i>
            </div>
            <div class="history-info">
                <div class="history-gesture">${item.gesture}</div>
                <div class="history-time">${formatTime(item.timestamp)}</div>
        `;
        
        // Add confidence information if available
        if (item.confidence !== null) {
            const confidencePercent = (item.confidence * 100).toFixed(1);
            historyHtml += `<div class="history-confidence">${confidencePercent}% confidence</div>`;
        }
        
        // Add latency information if available
        if (item.latency !== null) {
            historyHtml += `<div class="history-latency">${item.latency}ms</div>`;
        }
        
        // Close the history-info div
        historyHtml += `</div>`;
        
        li.innerHTML = historyHtml;
        historyList.appendChild(li);
    });
    
    // Scroll to the top to show the latest entry
    historyList.scrollTop = 0;
}

/**
 * Handle WebSocket message event
 */
function handleSocketMessage(event) {
    try {
        // Check if the message is binary (not used in this implementation)
        if (event.data instanceof Blob) {
            console.log('Received binary data from server');
            return;
        }
        
        // Parse the JSON message
        emojiDisplay.innerHTML = String(event.data); // Clear previous content
        const data = JSON.parse(event.data);
        console.log('Received message from server:', data);
        // Check if the data contains prediction information directly
        if (data.prediction !== undefined) {
            // This is the new format with prediction, confidence, latency, and debug_info
            console.log('Received gesture prediction:', data.prediction, 'with confidence:', data.confidence);
            
            // Display the gesture using the prediction data
            displayGesture(data);
            
            // Update performance stats if latency is available
            if (data.latency !== undefined) {
                updatePerformanceStats({
                    latency: data.latency,
                    // Use any additional stats from debug_info if available
                    fps: data.debug_info?.fps,
                    resolution: data.debug_info?.resolution
                });
            }
            
            return;
        }
        
        // Handle the previous message format if needed
        switch (data.type) {
            case "connected":
                // Connection acknowledgment
                console.log('Server acknowledged connection');
                break;
                
            case "info":
                // Process gesture information from the server - old format
                if (data.gesture) {
                    displayGesture(data.gesture);
                }
                
                // Update additional info if available
                if (data.stats) {
                    updatePerformanceStats(data.stats);
                }
                break;
                
            default:
                console.log('Received unknown message type:', data.type);
        }
    } catch (error) {
        console.error('Error processing WebSocket message:', error);
    }
}
function handleSocketOpen(event) {
  console.log('WebSocket connection established');
  isConnected = true;
  reconnectAttempts = 0;
  
  // Update the UI to reflect connection status
  updateConnectionStatusUI(true);
  
  // Start video capture after connection is established
  initializeVideoCapture()
    .then(success => {
      if (!success) {
        closeWebSocketConnection();
      }
    });
  
  // Send initial setup message if needed
  const setupMessage = {
    type: "setup",
    clientInfo: {
      browser: navigator.userAgent,
      screenSize: `${window.innerWidth}x${window.innerHeight}`,
      timestamp: new Date().toISOString()
    }
  };
  
  sendMessage(setupMessage);
}

/**
 * Handle WebSocket message event
 */

/**
 * Handle WebSocket close event
 */
function handleSocketClose(event) {
  console.log('WebSocket connection closed:', event.code, event.reason);
  isConnected = false;
  
  // Stop capturing frames
  stopCapturingFrames();
  
  // Update UI to reflect disconnection
  updateConnectionStatusUI(false);
  
  // Attempt to reconnect if the detection is still active
  if (videoPlaceholder.classList.contains('video-active') && reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
    console.log(`Attempting to reconnect (${reconnectAttempts + 1}/${MAX_RECONNECT_ATTEMPTS})...`);
    setTimeout(initializeWebSocketConnection, RECONNECT_DELAY);
    reconnectAttempts++;
  } else if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
    handleConnectionFailure('Failed to reconnect after multiple attempts.');
  }
}

/**
 * Handle WebSocket error event
 */
function handleSocketError(error) {
  console.error('WebSocket error:', error);
  
  // The connection will automatically close after an error,
  // which will trigger the onclose handler
}

/**
 * Update connection status in the UI
 */
function updateConnectionStatusUI(connected) {
  if (connected) {
    // Update legend box if it exists
    if (legendBox) {
      const modelValueElement = legendBox.querySelector('.legend-item:nth-child(1) .legend-value');
      if (modelValueElement) modelValueElement.textContent = 'HandNet-v2 (Connected)';
    }
  } else {
    // Update UI to show disconnected status
    if (legendBox) {
      const modelValueElement = legendBox.querySelector('.legend-item:nth-child(1) .legend-value');
      if (modelValueElement) modelValueElement.textContent = 'HandNet-v2 (Disconnected)';
    }
  }
}

/**
 * Update performance stats in the legend box
 */
function updatePerformanceStats(stats) {
  if (!legendBox) return;
  
  if (stats.latency) {
    const latencyValueElement = legendBox.querySelector('.legend-item:nth-child(2) .legend-value');
    if (latencyValueElement) latencyValueElement.textContent = `${stats.latency} ms`;
  }
  
  if (stats.fps) {
    const fpsValueElement = legendBox.querySelector('.legend-item:nth-child(3) .legend-value');
    if (fpsValueElement) fpsValueElement.textContent = stats.fps;
  }
  
  if (stats.resolution) {
    const resolutionValueElement = legendBox.querySelector('.legend-item:nth-child(4) .legend-value');
    if (resolutionValueElement) resolutionValueElement.textContent = stats.resolution;
  }
}

/**
 * Send message through WebSocket
 */
function sendMessage(message) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(message));
  } else {
    console.warn('Cannot send message, WebSocket is not connected');
  }
}

/**
 * Close WebSocket connection
 */
function closeWebSocketConnection() {
  // Stop capturing frames before closing the connection
  stopCapturingFrames();
  
  if (socket) {
    console.log('Closing WebSocket connection...');
    socket.close(1000, 'User stopped detection');
    isConnected = false;
  }
}

/**
 * Handle connection failure
 */
function handleConnectionFailure(message) {
  console.error(message);
  
  // Stop capturing frames
  stopCapturingFrames();
  
  // Reset the UI if currently in active state
  if (videoPlaceholder.classList.contains('video-active')) {
    videoPlaceholder.classList.remove('video-active');
    if (startButton) startButton.textContent = 'Start Detection';
    if (legendBox) legendBox.classList.remove('active');
    
    // Reset emoji display
    
    // Show an error message to the user
  }
}

/**
 * Generate a random session ID if not provided
 */
function generateSessionId() {
  return 'ms_' + Math.random().toString(36).substring(2, 15);
}

/**
 * Send current client status to the server
 */
function sendClientStatus(isActive) {
  const statusMessage = {
    type: "status",
    status: isActive ? "active" : "inactive",
    timestamp: new Date().toISOString()
  };
  
  sendMessage(statusMessage);
}

// Export functions that might be called from po.html
window.motionSpeakWS = {
  connect: initializeWebSocketConnection,
  disconnect: closeWebSocketConnection,
  isConnected: () => isConnected,
  sendStatus: sendClientStatus,
  startCapture: startCapturingFrames,
  stopCapture: stopCapturingFrames
};

// Clean up connection when the page unloads
window.addEventListener('beforeunload', () => {
  stopCapturingFrames();
  closeWebSocketConnection();
});
    // Gesture history array











    // Function to format the time nicely
    function formatTime(date) {
      const hours = date.getHours().toString().padStart(2, '0');
      const minutes = date.getMinutes().toString().padStart(2, '0');
      const seconds = date.getSeconds().toString().padStart(2, '0');
      const milliseconds = date.getMilliseconds().toString().padStart(3, '0');
      return `${hours}:${minutes}:${seconds}.${milliseconds}`;
    }
    
    // Function to display a gesture in the emoji display
    // Function to add a gesture to history
    // Function to update the history list in the UI
    // Check for mobile devices
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    
    // Start button click handler
    // Sidebar toggle handler
    sidebarToggle.addEventListener('click', function() {
      sidebarContainer.classList.toggle('open');
      
      if (sidebarContainer.classList.contains('open')) {
        toggleIcon.classList.remove('fa-chevron-right');
        toggleIcon.classList.add('fa-chevron-left');
      } else {
        toggleIcon.classList.remove('fa-chevron-left');
        toggleIcon.classList.add('fa-chevron-right');
      }
    });
    
    // History panel toggle handler
    historyToggle.addEventListener('click', function() {
      historyPanel.classList.toggle('open');
      
      if (historyPanel.classList.contains('open')) {
        historyToggleIcon.classList.remove('fa-chevron-right');
        historyToggleIcon.classList.add('fa-chevron-left');
      } else {
        historyToggleIcon.classList.remove('fa-chevron-left');
        historyToggleIcon.classList.add('fa-chevron-right');
      }
    });
    
    // Initialize UI based on screen size
    window.addEventListener('DOMContentLoaded', () => {
      if (isMobile) {
        // Mobile layout
        toggleIcon.classList.remove('fa-chevron-left');
        toggleIcon.classList.add('fa-chevron-down');
        historyToggleIcon.classList.remove('fa-chevron-left', 'fa-chevron-right');
        historyToggleIcon.classList.add('fa-chevron-up');
      } else {
        // Desktop layout - sidebar visible by default
        sidebarContainer.classList.add('open');
        historyPanel.classList.add('open');
      }
      
      // Initialize history list
      updateHistoryList();
    });
    
    // Handle window resize
    window.addEventListener('resize', () => {
      if (window.innerWidth <= 768) {
        // Mobile layout
        toggleIcon.classList.remove('fa-chevron-left', 'fa-chevron-right');
        historyToggleIcon.classList.remove('fa-chevron-left', 'fa-chevron-right');
        
        if (sidebarContainer.classList.contains('open')) {
          toggleIcon.classList.add('fa-chevron-down');
        } else {
          toggleIcon.classList.add('fa-chevron-up');
        }
        
        if (historyPanel.classList.contains('open')) {
          historyToggleIcon.classList.add('fa-chevron-down');
        } else {
          historyToggleIcon.classList.add('fa-chevron-up');
        }
      } else {
        // Desktop layout
        toggleIcon.classList.remove('fa-chevron-up', 'fa-chevron-down');
        historyToggleIcon.classList.remove('fa-chevron-up', 'fa-chevron-down');
        
        if (sidebarContainer.classList.contains('open')) {
          toggleIcon.classList.add('fa-chevron-left');
        } else {
          toggleIcon.classList.add('fa-chevron-right');
        }
        
        if (historyPanel.classList.contains('open')) {
          historyToggleIcon.classList.add('fa-chevron-left');
        } else {
          historyToggleIcon.classList.add('fa-chevron-right');
        }
      }
    });

    // Click on gesture items to preview them in the emoji display
    document.querySelectorAll('.gesture-item').forEach(item => {
      item.addEventListener('click', function() {
        if (videoPlaceholder.classList.contains('video-active')) {
          const gestureName = this.getAttribute('data-gesture');
          displayGesture(gestureName);
          
          // Clear any existing demo interval
          if (window.demoInterval) {
            clearInterval(window.demoInterval);
          }
        }
      });
    });

    // Support for dark/light mode
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      document.documentElement.classList.add('dark');
    }
    
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', event => {
      if (event.matches) {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
    });