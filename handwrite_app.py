import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import serial
import serial.tools.list_ports
from PIL import Image, ImageDraw, ImageTk
import threading
import time

class HandWriteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HandWriteDemo V1.0")
        self.root.geometry("800x450")
        self.root.minsize(800, 450)
        
        # Handwriting related variables
        self.big_canvas_size = 280
        self.small_canvas_size = 28
        self.big_image = Image.new('RGB', (self.big_canvas_size, self.big_canvas_size), 'white')
        self.small_image = Image.new('RGB', (self.small_canvas_size, self.small_canvas_size), 'black')
        self.big_draw = ImageDraw.Draw(self.big_image)
        self.small_draw = ImageDraw.Draw(self.small_image)
        
        self.current_x = 0
        self.current_y = 0
        self.is_drawing = False
        self.pic_byte = [0] * (28 * 28)
        
        # Serial port related variables
        self.serial_port = None
        self.serial_thread = None
        self.serial_running = False

        # ---- NEW: timing state ----
        self.last_send_time = None  # for PC round-trip latency
        self.waiting_for_response = False  # True after Send, False after first reply

        self.setup_ui()
        self.setup_canvas()
        self.update_big_canvas()
        self.update_small_image()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side handwriting area
        write_frame = ttk.LabelFrame(main_frame, text="WriteRegion")
        write_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Large canvas
        self.big_canvas = tk.Canvas(write_frame, width=self.big_canvas_size, 
                                  height=self.big_canvas_size, bg='white')
        self.big_canvas.pack(pady=10)
        
        # Button and preview area
        button_frame = ttk.Frame(write_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_picture)
        self.clear_btn.pack(side=tk.LEFT)
        
        # Small preview canvas
        self.small_canvas = tk.Canvas(button_frame, width=self.small_canvas_size, 
                                    height=self.small_canvas_size, bg='black')
        self.small_canvas.pack(side=tk.RIGHT)
        
        # Right side serial control area
        serial_frame = ttk.LabelFrame(main_frame, text="Serial")
        serial_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Serial control
        control_frame = ttk.Frame(serial_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(control_frame, text="COM:").pack(side=tk.LEFT)
        self.com_combo = ttk.Combobox(control_frame, width=10)
        self.com_combo.pack(side=tk.LEFT, padx=(5, 10))
        self.com_combo.bind('<Button-1>', self.refresh_com_ports)
        
        ttk.Label(control_frame, text="Baud:").pack(side=tk.LEFT)
        self.baud_entry = ttk.Entry(control_frame, width=10)
        self.baud_entry.pack(side=tk.LEFT, padx=(5, 10))
        self.baud_entry.insert(0, "115200")
        
        self.open_btn = ttk.Button(control_frame, text="Open", command=self.toggle_serial)
        self.open_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.send_btn = ttk.Button(control_frame, text="Send", command=self.send_data)
        self.send_btn.pack(side=tk.LEFT)
        
        # Log area
        log_frame = ttk.Frame(serial_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, font=('Consolas', 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize COM port list
        self.refresh_com_ports()
        
    def setup_canvas(self):
        # Bind mouse events
        self.big_canvas.bind('<Button-1>', self.on_mouse_down)
        self.big_canvas.bind('<B1-Motion>', self.on_mouse_move)
        self.big_canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        
    def refresh_com_ports(self, event=None):
        """Refresh COM port list"""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.com_combo['values'] = ports
        if ports and not self.com_combo.get():
            self.com_combo.set(ports[0])
            
    def toggle_serial(self):
        """Open/Close serial port"""
        try:
            if self.open_btn['text'] == "Open":
                port = self.com_combo.get()
                baud = int(self.baud_entry.get())
                
                if not port:
                    messagebox.showerror("Error", "Please select a COM port")
                    return
                    
                self.serial_port = serial.Serial(port, baud, timeout=1)
                self.serial_running = True
                self.serial_thread = threading.Thread(target=self.serial_read_thread)
                self.serial_thread.daemon = True
                self.serial_thread.start()
                
                self.open_btn['text'] = "Close"
                self.log_text.insert(tk.END, f"Serial port {port} opened at {baud} baud\n")
                
            else:
                self.serial_running = False
                if self.serial_port:
                    self.serial_port.close()
                    self.serial_port = None
                self.open_btn['text'] = "Open"
                self.log_text.insert(tk.END, "Serial port closed\n")
                
        except Exception as ex:
            messagebox.showerror("Error", f"Serial open/close error: {str(ex)}")

    def serial_read_thread(self):
        """Serial read thread"""
        while self.serial_running and self.serial_port:
            try:
                if self.serial_port.in_waiting:
                    line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        # 1) Measure round-trip latency (PC → MCU → PC)
                        if self.waiting_for_response and self.last_send_time is not None:
                            elapsed_ms = (time.perf_counter() - self.last_send_time) * 1000.0
                            self.waiting_for_response = False
                            msg = f"PC round-trip latency: {elapsed_ms:.2f} ms\n"
                            self.root.after(0, lambda m=msg: self.log_text.insert(tk.END, m))

                        # 2) Parse MCU inference time (pure computation) from UART text
                        mcu_time = self.extract_mcu_inference_time(line)
                        if mcu_time is not None:
                            msg = f"STM32 inference time: {mcu_time} ms\n"
                            self.root.after(0, lambda m=msg: self.log_text.insert(tk.END, m))

                        # 3) Log raw line
                        self.root.after(0, lambda l=line: self.log_text.insert(tk.END, f"{l}\n"))
                        self.root.after(0, lambda: self.log_text.see(tk.END))

            except Exception as ex:
                # Capture message now, so 'ex' isn't needed later
                err_msg = f"Serial read error: {str(ex)}"
                self.root.after(0, lambda m=err_msg: messagebox.showerror("Error", m))
                break

    def send_data(self):
        """Send handwriting data"""
        try:
            if not self.serial_port or not self.serial_port.is_open:
                messagebox.showerror("Error", "Serial port is not open")
                return
                
            self.log_text.delete(1.0, tk.END)
            
            # Prepare data packet
            cmd_head = [0xAA]
            cmd_end = [0x0D, 0x0A]

            # Start round-trip timer
            self.last_send_time = time.perf_counter()
            self.waiting_for_response = True
            
            # Send data
            self.serial_port.write(bytes(cmd_head))
            self.serial_port.write(bytes(self.pic_byte))
            self.serial_port.write(bytes(cmd_end))
            
            self.log_text.insert(tk.END, "Data sent successfully\n")
            
        except Exception as ex:
            messagebox.showerror("Error", f"Serial send error: {str(ex)}")
            
    def on_mouse_down(self, event):
        """Mouse down event"""
        self.is_drawing = True
        self.current_x = event.x
        self.current_y = event.y
        # Draw a point
        r = 12
        self.big_canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill='black', outline='black')
        self.big_draw.ellipse([event.x - r, event.y - r, event.x + r, event.y + r], fill='black')
        
    def on_mouse_move(self, event):
        """Mouse move event"""
        if self.is_drawing:
            # Draw on large canvas
            r = 12
            self.big_canvas.create_line(self.current_x, self.current_y, event.x, event.y, fill='black', width=25, capstyle=tk.ROUND, smooth=True)
            self.big_draw.line([self.current_x, self.current_y, event.x, event.y], fill='black', width=25)
            self.current_x = event.x
            self.current_y = event.y
        
    def on_mouse_up(self, event):
        """Mouse up event"""
        self.is_drawing = False
        self.update_big_canvas()
        self.update_small_image()
        
    def update_big_canvas(self):
        """Update large canvas display"""
        # Use PIL.ImageTk.PhotoImage to display PIL image
        self.big_photo = ImageTk.PhotoImage(self.big_image)
        self.big_canvas.create_image(0, 0, image=self.big_photo, anchor=tk.NW)
        
    def update_small_image(self):
        """Update small image and byte array"""
        # Reset small image
        self.small_image = Image.new('RGB', (self.small_canvas_size, self.small_canvas_size), 'black')
        self.small_draw = ImageDraw.Draw(self.small_image)
        
        # Reset byte array
        self.pic_byte = [0] * (28 * 28)
        
        # Sample large image and update small image
        for y in range(28):
            for x in range(28):
                # Sample from large image (sample one point every 10x10 pixels)
                sample_x = x * 10
                sample_y = y * 10
                
                if sample_x < self.big_canvas_size and sample_y < self.big_canvas_size:
                    pixel = self.big_image.getpixel((sample_x, sample_y))
                    if pixel == (0, 0, 0):  # Black pixel
                        self.small_draw.point((x, y), fill='white')
                        self.pic_byte[y * 28 + x] = 1
        # Update small canvas display
        self.small_photo = ImageTk.PhotoImage(self.small_image)
        self.small_canvas.create_image(0, 0, image=self.small_photo, anchor=tk.NW)
        
    def clear_picture(self):
        """Clear canvas"""
        # Clear large canvas
        self.big_image = Image.new('RGB', (self.big_canvas_size, self.big_canvas_size), 'white')
        self.big_draw = ImageDraw.Draw(self.big_image)
        self.big_canvas.delete('all')
        self.update_big_canvas()
        
        # Clear small canvas
        self.small_image = Image.new('RGB', (self.small_canvas_size, self.small_canvas_size), 'black')
        self.small_draw = ImageDraw.Draw(self.small_image)
        self.small_photo = ImageTk.PhotoImage(self.small_image)
        self.small_canvas.create_image(0, 0, image=self.small_photo, anchor=tk.NW)
        
        # Reset byte array
        self.pic_byte = [0] * (28 * 28)

    def extract_mcu_inference_time(self, line: str):
        """
        Parse MCU UART output for inference time message.
        Expected format examples:
            'Inference time: 5 ms'
            'Inference: 12 ms'
            'Inference time = 7ms'
        Returns the integer number of ms, or None.
        """
        line = line.lower()
        if "inference" in line and "ms" in line:
            try:
                # Extract digits before 'ms'
                import re
                match = re.search(r"(\d+)\s*ms", line)
                if match:
                    return int(match.group(1))
            except:
                return None
        return None


def main():
    root = tk.Tk()
    app = HandWriteApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 