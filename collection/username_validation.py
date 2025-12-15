import tkinter as tk 
from tkinter import ttk 
import re 
 
class UsernameValidatorApp: 
    def __init__(self, root): 
        self.root = root 
        self.root.title("Automata Theory: Regex Filter System") 
        self.root.geometry("500x350") 
        self.root.configure(bg="#f0f2f5") 
 
        # --- THEORETICAL DEFINITION --- 
        # The Regex Pattern representing the Regular Language L: 
        # ^             : Start of string 
        # [a-zA-Z]      : Initial State (must be a letter) 
        # [a-zA-Z0-9_]  : Intermediate State transitions (alphanumeric + _) 
        # {3,13}        : Repeats previous set (adjusting for start/end char to equal 5-15 total) 
        # [a-zA-Z0-9]   : Final Acceptance State (must not end with underscore) 
        # $             : End of string 
        self.pattern = r"^[a-zA-Z][a-zA-Z0-9_]{3,13}[a-zA-Z0-9]$" 
         
        # Pre-compile the regex (Optimization similar to DFA construction) 
        self.regex_compiler = re.compile(self.pattern) 
 
        self.setup_ui() 
 
    def setup_ui(self): 
        # Header 
        header = tk.Label( 
            self.root,  
            text="Username Syntax Validator",  
            font=("Helvetica", 18, "bold"), 
            bg="#f0f2f5", fg="#333" 
        ) 
        header.pack(pady=30) 
 
        # Instruction Frame 
        info_frame = tk.Frame(self.root, bg="white", padx=10, pady=10) 
        info_frame.pack(pady=10) 
         
        rules = ( 
            "RULES:\n" 
            "1. Must start with a Letter.\n" 
            "2. Length: 5 - 15 characters.\n" 
            "3. Allowed: Alphanumeric & Underscore.\n" 
            "4. Cannot end with an Underscore." 
        ) 
        tk.Label(info_frame, text=rules, bg="white", justify="left", fg="#555").pack() 
 
        # Input Field 
        self.username_var = tk.StringVar() 
        self.entry = ttk.Entry( 
            self.root,  
            textvariable=self.username_var,  
            font=("Courier", 14),  
            width=25 
        ) 
        self.entry.pack(pady=20) 
         
        # Bind the 'KeyRelease' event to our validator function 
        # This simulates the transition of states in an automata as inputs are consumed. 
        self.entry.bind("<KeyRelease>", self.validate_input) 
 
        # Status Indicator 
        self.status_label = tk.Label( 
            self.root,  
            text="Waiting for input...",  
            font=("Helvetica", 12, "bold"), 
            bg="#f0f2f5", fg="#777" 
        ) 
        self.status_label.pack(pady=10) 
 
    def validate_input(self, event=None): 
        current_text = self.username_var.get() 
         
        # Edge case: Empty string 
        if not current_text: 
            self.status_label.config(text="Waiting for input...", fg="#777") 
            self.entry.config(foreground="black") 
            return 
 
        # --- THE VALIDATION (Simulating the Machine) --- 
        if self.regex_compiler.match(current_text): 
            # ACCEPT STATE 
            self.status_label.config(text="   Valid Username", fg="#2ecc71") # Green 
            self.entry.config(foreground="#2ecc71") 
        else: 
            # REJECT STATE (Trap State) 
            self.analyze_error(current_text) 
            self.entry.config(foreground="#e74c3c") # Red 
 
    def analyze_error(self, text): 
        """ 
        Provides detailed feedback on why the string was rejected, 
        effectively describing which 'state' the automata failed at. 
        """ 
        msg = "✘ Invalid: " 
        if len(text) < 5: 
            msg += "Too short (min 5)" 
        elif len(text) > 15: 
            msg += "Too long (max 15)" 
        elif not text[0].isalpha(): 
            msg += "Must start with a letter" 
        elif text[-1] == "_": 
            msg += "Cannot end with '_'" 
        elif not re.match(r"^[a-zA-Z0-9_]*$", text): 
            msg += "Illegal characters found" 
        else: 
            msg += "Syntax error" 
             
        self.status_label.config(text=msg, fg="#e74c3c") 
 
if __name__ == "__main__": 
    root = tk.Tk() 
    app = UsernameValidatorApp(root) 
    root.mainloop()