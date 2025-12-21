import numpy as np
import cv2
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ============================================================================
# FUZZY LOGIC SYSTEM - Simple Membership Functions
# ============================================================================

class FuzzyMembership:
    """Simple fuzzy membership functions for fuzzification"""
    
    @staticmethod
    def triangular(x, a, b, c):
        """
        Triangular membership function
        a: left point (membership = 0)
        b: peak point (membership = 1)
        c: right point (membership = 0)
        """
        if x < a or x > c:
            return 0
        elif x == b:
            return 1
        elif a < x < b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)
    
    @staticmethod
    def trapezoidal(x, a, b, c, d):
        """
        Trapezoidal membership function
        a: left point (membership = 0)
        b: left plateau point (membership = 1)
        c: right plateau point (membership = 1)
        d: right point (membership = 0)
        """
        if x < a or x > d:
            return 0
        elif a <= x < b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return 1
        else:  # c < x <= d
            return (d - x) / (d - c)
    
    @staticmethod
    def gaussian(x, mean, sigma):
        """Gaussian membership function"""
        return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


# ============================================================================
# SIMPLE PERCEPTRON
# ============================================================================

class SimplePerceptron:
    """
    A simple single-layer perceptron for binary classification
    Uses fuzzy inputs for soft decision making
    """
    
    def __init__(self, input_size, learning_rate=0.01):
        """
        Initialize perceptron
        input_size: number of features
        learning_rate: how fast the perceptron learns
        """
        # Initialize weights randomly (small values)
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0
        self.learning_rate = learning_rate
    
    def sigmoid(self, x):
        """Sigmoid activation function (smooth, non-linear)"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def predict(self, X):
        """
        Make prediction
        X: input features (can be 1D or 2D array)
        Returns: probability between 0 and 1
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Linear combination: z = w·x + b
        z = np.dot(X, self.weights) + self.bias
        
        # Apply sigmoid activation
        return self.sigmoid(z)
    
    def train(self, X, y, epochs=100):
        """
        Train the perceptron using gradient descent
        X: training features (samples, features)
        y: training labels (0 or 1)
        epochs: number of training iterations
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        history = []
        
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(len(X)):
                # Forward pass
                prediction = self.predict(X[i])
                target = y[i]
                
                # Calculate error
                error = target - prediction
                total_loss += error ** 2
                
                # Backward pass (gradient descent)
                delta = error * prediction * (1 - prediction)
                self.weights += self.learning_rate * delta * X[i]
                self.bias += self.learning_rate * delta
            
            # Track average loss
            avg_loss = total_loss / len(X)
            history.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {float(avg_loss):.4f}")
        
        return history


# ============================================================================
# FUZZY PERCEPTRON HYBRID - Combine both systems
# ============================================================================

class FuzzyPerceptron:
    """
    Hybrid system that combines:
    1. Fuzzy logic for feature preprocessing (fuzzification)
    2. Perceptron for learning and classification
    """
    
    def __init__(self, input_size, learning_rate=0.01):
        self.perceptron = SimplePerceptron(input_size, learning_rate)
        self.membership_functions = {}
    
    def set_fuzzy_feature(self, feature_name, membership_func):
        """Define fuzzy membership for a feature"""
        self.membership_functions[feature_name] = membership_func
    
    def fuzzify(self, raw_input):
        """Convert raw input to fuzzy values [0, 1]"""
        fuzzy_input = []
        for feature_name, value in raw_input.items():
            if feature_name in self.membership_functions:
                fuzzy_value = self.membership_functions[feature_name](value)
                fuzzy_input.append(fuzzy_value)
        return np.array(fuzzy_input)
    
    def predict(self, raw_input):
        """Make prediction from raw input"""
        fuzzy_input = self.fuzzify(raw_input)
        return self.perceptron.predict(fuzzy_input)


# ============================================================================
# OPENCV INTEGRATION - Live Camera Processing
# ============================================================================

class FuzzyVisionAnalyzer:
    """Analyzes live video feed using fuzzy logic and perceptron"""
    
    def __init__(self, model):
        """
        Initialize with a trained FuzzyPerceptron model
        model: FuzzyPerceptron instance
        """
        self.model = model
        self.cap = None
        self.running = False
    
    def extract_features(self, frame):
        """
        Extract visual features from frame
        Returns: dict with brightness, motion, color intensity, and faces
        """
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness (normalized 0-1)
        brightness = np.mean(gray) / 255.0
        
        # Calculate color saturation (normalized 0-1)
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        
        # Calculate motion by comparing with previous frame (edge detection)
        edges = cv2.Canny(gray, 50, 150)
        motion = np.sum(edges) / (frame.shape[0] * frame.shape[1] * 255.0)
        
        # Detect faces for presence indicator
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_detected = 1.0 if len(faces) > 0 else 0.0
        
        return {
            'brightness': brightness,
            'saturation': saturation,
            'motion': motion,
            'face_detected': face_detected,
            'faces': faces  # Return face coordinates for drawing
        }
    
    def run_live_analysis(self):
        """Run real-time analysis on webcam feed"""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("\n" + "=" * 60)
        print("FUZZY LOGIC LIVE CAMERA ANALYSIS")
        print("=" * 60)
        print("Press 'q' to quit, 's' to take screenshot")
        print("-" * 60 + "\n")
        
        self.running = True
        frame_count = 0
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Resize for faster processing
            frame = cv2.resize(frame, (640, 480))
            frame_count += 1
            
            # Extract features
            features = self.extract_features(frame)
            
            # Draw bounding boxes around detected faces
            for (x, y, w, h) in features['faces']:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Make prediction using fuzzy perceptron
            prediction = self.model.predict(features)[0]
            
            # Determine status
            if prediction < 0.33:
                status = "DARK"
                color = (0, 0, 255)  # Red
            elif prediction < 0.66:
                status = "NORMAL"
                color = (0, 255, 0)  # Green
            else:
                status = "BRIGHT"
                color = (0, 255, 255)  # Yellow
            
            # Draw information on frame
            cv2.putText(frame, f"Status: {status}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Confidence: {prediction:.2%}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw feature values
            y_offset = 130
            cv2.putText(frame, f"Brightness: {features['brightness']:.2f}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Saturation: {features['saturation']:.2f}", 
                       (20, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Motion: {features['motion']:.2f}", 
                       (20, y_offset + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            face_text = "Yes" if features['face_detected'] > 0.5 else "No"
            cv2.putText(frame, f"Face Detected: {face_text}", 
                       (20, y_offset + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw frame count
            cv2.putText(frame, f"Frame: {frame_count}", (20, 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Display frame
            cv2.imshow('Fuzzy Logic Vision Analyzer', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f'screenshot_{frame_count}.png'
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Camera analysis complete!")


# ============================================================================
# EXAMPLE: OpenCV Live Camera Vision Analysis
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FUZZY LOGIC - OPENCV LIVE CAMERA ANALYSIS")
    print("=" * 60)
    
    # Create hybrid fuzzy perceptron system
    print("\n1. INITIALIZING HYBRID FUZZY PERCEPTRON")
    print("-" * 60)
    
    hybrid = FuzzyPerceptron(input_size=4, learning_rate=0.1)
    
    # Define fuzzy membership functions for visual features
    hybrid.set_fuzzy_feature('brightness', 
        lambda x: FuzzyMembership.triangular(x, 0, 0.5, 1.0))
    hybrid.set_fuzzy_feature('saturation',
        lambda x: FuzzyMembership.gaussian(x, 0.5, 0.3))
    hybrid.set_fuzzy_feature('motion',
        lambda x: FuzzyMembership.triangular(x, 0, 0.02, 0.1))
    hybrid.set_fuzzy_feature('face_detected',
        lambda x: FuzzyMembership.triangular(x, 0, 0.5, 1.0))
    
    # Prepare training data for vision classification
    # (brightness, saturation, motion, face_detected) -> classification
    print("Training on vision data...")
    X_raw = np.array([
        {'brightness': 0.2, 'saturation': 0.3, 'motion': 0.01, 'face_detected': 0},  # dark
        {'brightness': 0.3, 'saturation': 0.4, 'motion': 0.015, 'face_detected': 0}, # dark
        {'brightness': 0.5, 'saturation': 0.5, 'motion': 0.02, 'face_detected': 1},  # normal
        {'brightness': 0.6, 'saturation': 0.6, 'motion': 0.025, 'face_detected': 1}, # normal
        {'brightness': 0.8, 'saturation': 0.7, 'motion': 0.05, 'face_detected': 1},  # bright
        {'brightness': 0.9, 'saturation': 0.8, 'motion': 0.08, 'face_detected': 1},  # bright
    ])
    
    y_train = np.array([0, 0, 1, 1, 2, 2]) / 2.0  # Normalize to [0, 1]
    
    # Fuzzify all training data
    X_fuzzified = np.array([hybrid.fuzzify(sample) for sample in X_raw])
    
    # Train the perceptron
    history = hybrid.perceptron.train(X_fuzzified, y_train, epochs=100)
    print("Training complete!")
    
    # Start live camera analysis
    print("\n2. STARTING LIVE CAMERA ANALYSIS")
    print("-" * 60)
    
    analyzer = FuzzyVisionAnalyzer(hybrid)
    analyzer.run_live_analysis()
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
