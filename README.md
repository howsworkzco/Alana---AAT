ALANA Advanced Autonomous Agent Blueprint
*A Comprehensive Guide to Building a Mobile, Multimodal Humanoid Personal Assistant*

---

## 1. Project Overview & System Architecture
This document serves as the master blueprint for transforming the standard, stationary ALANA humanoid robot into a fully autonomous, mobile, smart-home integrated personal assistant. 

Unlike reactive chatbots, this architecture implements an advanced **Sense-Plan-Act** cycle driven by a continuous **Self-Prompting Loop** based on the MACHINA1 project principles. Processing is distributed across a main computational host (external PC moving to a Raspberry Pi 5) and localized microcontrollers managing hardware constraints.

```
                    ┌─────────────────────────┐
                    │  Master Python Agent    │
                    │       (Host PC)         │
                    └────▲───────────────┬────┘
                         │               │
           ┌─────────────┴─┐           ┌─▼──────────────┐
           │ Sensor Inputs │           │ Llama 3 Brain  │
           └─▲───────────▲─┘           └─┬──────────────┘
             │           │               │
       [YOLO / VLM]   [Voice]            ▼
                                   [Tool Execution] 
                               (Serial Commands to ESP)
```

---

## 2. Master Bill of Materials (BOM) & Budget Estimations

| Sub-System | Component | Est. Cost (USD) | Purpose |
| :--- | :--- | :--- | :--- |
| **Mechanical & Body** | 3kg PETG/Carbon Filament + structural PVC pipes | $40 – $60 | Printing the full torso, dual 6-DoF arms, and custom head. |
| **Main Compute** | Raspberry Pi 5 (8GB) or External Prototyping PC | $80 – $100 | Runs the master Python event loop, YOLOv8, and API orchestrations. |
| **Microcontroller** | ESP32 Dev Board + 2x PCA9685 16-Ch Servo Drivers | $25 – $35 | Decouples PWM logic and data lines from the main compute host. |
| **Actuators** | 12x MG996R Servos (Arms) + 10x SG90 Micro-Servos (Fingers) | $85 – $110 | Moves the body limbs and provides 5-finger articulated dexterity. |
| **Mobility Platform** | 4WD Mecanum Wheel Chassis + Encoder DC Motors + Drive Board | $50 – $85 | Provides omnidirectional movement, strafing, and position tracking. |
| **Sensors & Audio** | Autofocus USB Camera Module + Microphone Array + Amp & Speaker | $50 – $75 | Facilitates multimodal sheet/text vision, spatial tracking, and sound. |
| **Power Delivery** | 12V 10A AC-to-DC PSU + Dual 5V 5A Buck Converters | $25 – $45 | Prevents brownouts by cleanly separating logic power from servo rails. |
| **Total Estimated Build Budget** | **Advanced Configuration** | **$355 – $510** | *Assumes access to a 3D printer and base modeling toolsets.* |

---

## 3. Structural & Mechanical Modifications
1. **The Hand (InMoov Variant):** Move away from standard single-tendon solid claws. Print a fully articulated hand featuring hollow tendon channels tracking down to the forearm. Use 0.5mm to 0.8mm braided PE fishing line for pulling joints. Add thin liquid silicone rubber pads to inner fingertips to generate high-friction contact points required for paper handling.
2. **Camera Alignment:** Mount an autofocus camera directly inside the brow or forehead section angled 15 degrees downward. Standard fixed-focus lenses will render text illegible at close document-viewing proximity.
3. **The Lower Base:** Reconfigure the central structural spine. Avoid fixing the base flange directly to tables. Construct a wide, low-gravity chassis frame to support the vertical PVC counterweights when arms fully extend with a payload.

---

## 4. Host PC Software Stack: Custom Python Code

### A. The Master Autonomous Event Loop (`agent_core.py`)
```python
import json
import time
from groq import Groq

AUTONOMOUS_SYSTEM_PROMPT = """
You are ALANA, a fully autonomous, life-size humanoid robot agent operating via a continuous Sense-Plan-Act cycle.
DIRECTIVES:
1. PATROL: Keep the room monitored. Execute scans if stationary for too long.
2. GREET: Address spotted humans warmly and politely.
3. SORT & ASSIST: If you spot a misplaced item or document worksheet, pick it up or process it.
OPERATIONAL RULES: Always execute function tools when required. Never output long chatty paragraphs to yourself.
"""

def run_self_prompting_loop(client, vision_system, hardware):
    messages = [
        {"role": "system", "content": AUTONOMOUS_SYSTEM_PROMPT},
        {"role": "user", "content": "System Boot Successful. ALANA active. Begin autonomous observation."}
    ]
    print("🚀 ALANA Autonomous Core Activated...")
    
    while True:
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                tools=tools_registry,
                tool_choice="auto"
            )
            assistant_msg = response.choices.message
            if assistant_msg.content:
                print(f"🧠 ALANA Thought: {assistant_msg.content}")
                messages.append({"role": "assistant", "content": assistant_msg.content})
            
            if assistant_msg.tool_calls:
                for tool_call in assistant_msg.tool_calls:
                    messages.append(assistant_msg)
                    result = execute_physical_tool(tool_call.function.name, json.loads(tool_call.function.arguments), vision_system, hardware)
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": result})
            else:
                time.sleep(2)
                messages.append({"role": "user", "content": "Heartbeat check. What is your next state update?"})
            if len(messages) > 12:
                messages = [messages[0]] + messages[-11:]
        except Exception as e:
            print(f"⚠️ Loop Error: {e}"); time.sleep(5)
```

### B. Multimodal Vision & OCR Engine (`vision_system.py`)
```python
import cv2
import base64
from ultralytics import YOLO

class AlanaVisionSystem:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        
    def scan_environment(self, cap):
        ret, frame = cap.read()
        if not ret: return "Error: Camera inaccessible."
        h, w, _ = frame.shape
        results = self.model(frame, verbose=False)
        detected = []
        for box in results[0].boxes:
            if box.conf[0] < 0.5: continue
            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
            zone = "left" if x_center < w/3 else "center" if x_center < 2*w/3 else "right"
            detected.append(f"a {self.model.names[int(box.cls[0])]} located on the {zone} side")
        return "Vision Scan Results: " + ", ".join(detected) if detected else "Room empty."

    def look_at_paper(self, client, user_question, cap):
        ret, frame = cap.read()
        if not ret: return "Error: Failed to freeze frame."
        _, buffer = cv2.imencode('.jpg', frame)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": f"Read and answer based on this document: {user_question}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}}
            ]}], max_tokens=300
        )
        return response.choices.message.content
```

### C. Spatial Serial Routing Engine (`hardware_bridge.py`)
```python
import serial
import time

class AlanaHardwareBridge:
    def __init__(self, port="COM3"):
        try:
            self.ser = serial.Serial(port, 115200, timeout=1)
            time.sleep(2)
        except Exception as e:
            self.ser = None
            print(f"Serial down: {e}")
            
    def send_raw_command(self, cmd_string):
        if not self.ser: return "Serial unconfigured."
        self.ser.write(f"{cmd_string}\n".encode('utf-8'))
        while True:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                if line == "DONE": return "Action completed successfully."
                if line.startswith("ERR:"): return f"Hardware Error: {line}"
```

---

## 5. Microcontroller Firmware: ESP32 Core Setup
Flash this logic to coordinates incoming commands and step instructions safely over the PCA9685 servo busses.

```cpp
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm1 = Adafruit_PWMServoDriver(0x40);
Adafruit_PWMServoDriver pwm2 = Adafruit_PWMServoDriver(0x41);

void setup() {
  Serial.begin(115200);
  pwm1.begin(); pwm1.setPWMFreq(50);
  pwm2.begin(); pwm2.setPWMFreq(50);
  Serial.println("MCU Boot Completed. Awaiting commands...");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.startsWith("ARM_R:")) {
      // Expected payload format: "ARM_R:SERVO_NUM,ANGLE_DEG"
      int separator = command.indexOf(',');
      int servoNum = command.substring(6, separator).toInt();
      int angle = command.substring(separator + 1).toInt();
      
      // Convert standard 0-180 degree angles to PCA9685 PWM pulse lengths
      int pulse = map(angle, 0, 180, 150, 600);
      pwm1.setPWM(servoNum, 0, pulse);
      
      delay(200);
      Serial.println("DONE");
    }
  }
}
```

---

## 6. Implementation Checklist & Operational Phases
* [ ] **Phase 1: Local Setup & Simulation:** Install Python, OpenCV, Ultralytics YOLO, and dependencies on your PC. Test your master agent loop using a standalone standard USB webcam. Verify Llama 3.2 Vision successfully analyzes test worksheets before printing physical frames.
* [ ] **Phase 2: Hand Manipulation Printing:** Print a single 5-finger tendon arm configuration. Connect the servos to a PCA9685 driver block. Connect the driver block to your ESP32. Confirm individual finger flexing commands function via Serial strings.
* [ ] **Phase 3: Frame Assembly & Integration:** Mount the structural torso directly over your weighted mobile Mecanum base. Ensure power distribution limits current draw evenly. Combine the independent code files into the final operational system script.
