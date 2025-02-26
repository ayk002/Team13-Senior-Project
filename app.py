import tkinter as tk
from tkinter import *
from tkinter import ttk, messagebox

class MyApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set window title and size
        self.title("Dysphagia Screener")
        self.geometry("1200x900")  # Width x Height

        # Configure menu bar
        self.create_menu()

        # Main Frame
        self.main_frame = ttk.Frame(self, padding=10)
        self.main_frame.pack(expand=True, fill="both")

        # Add widgets here  
        self.create_widgets()

        # Status Bar
        self.status = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN, anchor="w")
        self.status_bar.pack(side="bottom", fill="x")

    def create_menu(self):
        """Create a menu bar for the application"""
        menubar = tk.Menu(self)

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New", command=self.dummy_action)
        file_menu.add_command(label="Open", command=self.dummy_action)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def create_widgets(self):
        """Create widgets inside the main frame"""
        ttk.Label(self.main_frame, text="Welcome to Tkinter GUI!").pack(pady=10)
        ttk.Button(self.main_frame, text="Click Me", command=self.on_button_click).pack(pady=5)

    def on_button_click(self):
        """Handle button click event"""
        self.status.set("Button Clicked!")
        messagebox.showinfo("Message", "You clicked the button!")

    def show_about(self):
        """Display About information"""
        messagebox.showinfo("About", "Tkinter GUI Application\nVersion 1.0\nCross-Platform Ready")

    def dummy_action(self):
        """Placeholder for menu actions"""
        messagebox.showinfo("Info", "This is a placeholder action")

if __name__ == "__main__":
    app = MyApp()
    app.mainloop()
