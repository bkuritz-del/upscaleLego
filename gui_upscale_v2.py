#!/usr/bin/env python3
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from upscale_core import UpscaleSettings, process_batch


class UpscaleGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LEGO Print Upscaler")
        self.root.geometry("920x760")

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()

        self.scale_percent_var = tk.StringVar(value="1027.636")
        self.dpi_var = tk.StringVar(value="300")
        self.bleed_var = tk.StringVar(value="1.0")

        self.bleed_mode_var = tk.StringVar(value="dominant")
        self.custom_bleed_color_var = tk.StringVar(value="0,0,0")

        self.resample_var = tk.StringVar(value="bicubic")
        self.two_step_var = tk.BooleanVar(value=True)
        self.sharpen_var = tk.BooleanVar(value=True)
        self.radius_var = tk.StringVar(value="0.8")
        self.percent_var = tk.StringVar(value="160")
        self.threshold_var = tk.StringVar(value="2")

        self.pad_square_var = tk.BooleanVar(value=True)
        self.background_mode_var = tk.StringVar(value="auto")
        self.custom_bg_color_var = tk.StringVar(value="255,255,255")

        self.output_format_var = tk.StringVar(value="same")
        self.suffix_var = tk.StringVar(value="_print")
        self.recursive_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=False)
        self.quality_var = tk.StringVar(value="92")

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        notebook = ttk.Notebook(main)
        notebook.pack(fill="both", expand=True)

        basic_tab = ttk.Frame(notebook, padding=12)
        processing_tab = ttk.Frame(notebook, padding=12)
        output_tab = ttk.Frame(notebook, padding=12)
        log_tab = ttk.Frame(notebook, padding=12)

        notebook.add(basic_tab, text="Basic")
        notebook.add(processing_tab, text="Processing")
        notebook.add(output_tab, text="Output")
        notebook.add(log_tab, text="Run Log")

        self._build_basic_tab(basic_tab)
        self._build_processing_tab(processing_tab)
        self._build_output_tab(output_tab)
        self._build_log_tab(log_tab)

        button_row = ttk.Frame(main)
        button_row.pack(fill="x", pady=(10, 0))

        self.progress = ttk.Progressbar(button_row, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.status_label = ttk.Label(button_row, text="Ready")
        self.status_label.pack(side="left", padx=(0, 10))

        self.run_button = ttk.Button(button_row, text="Run Upscale", command=self.run_clicked)
        self.run_button.pack(side="right")

    def _build_basic_tab(self, parent):
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text="Input Folder").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(parent, text="Browse", command=self.browse_input).grid(row=0, column=2, pady=4)

        ttk.Label(parent, text="Output Folder").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(parent, text="Browse", command=self.browse_output).grid(row=1, column=2, pady=4)

        ttk.Label(parent, text="Scale Percent").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.scale_percent_var, width=12).grid(row=2, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(parent, text="DPI").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Combobox(parent, textvariable=self.dpi_var, values=["150", "300", "600"], state="readonly", width=10).grid(row=3, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(parent, text="Bleed (inches each side)").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Combobox(parent, textvariable=self.bleed_var, values=["0", "0.125", "0.25", "0.5", "1.0"], width=10).grid(row=4, column=1, sticky="w", padx=6, pady=4)

        ttk.Checkbutton(parent, text="Process subfolders recursively", variable=self.recursive_var).grid(row=5, column=0, columnspan=2, sticky="w", pady=6)
        ttk.Checkbutton(parent, text="Pad image to square before scaling", variable=self.pad_square_var).grid(row=6, column=0, columnspan=2, sticky="w", pady=6)

    def _build_processing_tab(self, parent):
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text="Upscaling Method").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(parent, textvariable=self.resample_var, values=["nearest", "bilinear", "bicubic", "lanczos"], state="readonly", width=15).grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Checkbutton(parent, text="Two-step upscale", variable=self.two_step_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=6)
        ttk.Checkbutton(parent, text="Sharpen edges", variable=self.sharpen_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=6)

        ttk.Label(parent, text="Unsharp Radius").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.radius_var, width=12).grid(row=3, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(parent, text="Unsharp Percent").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.percent_var, width=12).grid(row=4, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(parent, text="Unsharp Threshold").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.threshold_var, width=12).grid(row=5, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(parent, text="Bleed Mode").grid(row=6, column=0, sticky="w", pady=4)
        ttk.Combobox(parent, textvariable=self.bleed_mode_var, values=["dominant", "edge", "mirror", "custom"], state="readonly", width=15).grid(row=6, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(parent, text="Custom Bleed Color (R,G,B)").grid(row=7, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.custom_bleed_color_var, width=18).grid(row=7, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(parent, text="Square Padding Background").grid(row=8, column=0, sticky="w", pady=4)
        ttk.Combobox(parent, textvariable=self.background_mode_var, values=["auto", "transparent", "white", "black", "custom"], state="readonly", width=15).grid(row=8, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(parent, text="Custom Padding Color (R,G,B)").grid(row=9, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.custom_bg_color_var, width=18).grid(row=9, column=1, sticky="w", padx=6, pady=4)

    def _build_output_tab(self, parent):
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text="Output Format").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(parent, textvariable=self.output_format_var, values=["same", "png", "jpg", "webp", "tif"], state="readonly", width=15).grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(parent, text="Filename Suffix").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.suffix_var, width=18).grid(row=1, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(parent, text="JPEG / WebP Quality").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.quality_var, width=12).grid(row=2, column=1, sticky="w", padx=6, pady=4)

        ttk.Checkbutton(parent, text="Overwrite existing files", variable=self.overwrite_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=6)

    def _build_log_tab(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        self.log_text = tk.Text(parent, wrap="word", height=20)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def browse_input(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_var.set(folder)

    def browse_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_var.set(folder)

    def log(self, message: str):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def parse_rgb(self, value: str) -> tuple[int, int, int] | None:
        value = value.strip()
        if not value:
            return None
        parts = [p.strip() for p in value.split(",")]
        if len(parts) != 3:
            raise ValueError("Color must be R,G,B")
        rgb = tuple(int(p) for p in parts)
        if any(c < 0 or c > 255 for c in rgb):
            raise ValueError("Color values must be between 0 and 255")
        return rgb

    def build_settings(self) -> UpscaleSettings:
        input_dir = Path(self.input_var.get().strip())
        output_dir = Path(self.output_var.get().strip())

        if not self.input_var.get().strip() or not self.output_var.get().strip():
            raise ValueError("Please choose both input and output folders.")

        scale_percent = float(self.scale_percent_var.get())
        if scale_percent <= 0:
            raise ValueError("Scale Percent must be greater than 0.")

        return UpscaleSettings(
            input_dir=input_dir,
            output_dir=output_dir,
            scale_percent=scale_percent,
            dpi=int(self.dpi_var.get()),
            bleed_inches=float(self.bleed_var.get()),
            bleed_mode=self.bleed_mode_var.get(),
            custom_bleed_color=self.parse_rgb(self.custom_bleed_color_var.get()),
            resample=self.resample_var.get(),
            two_step=self.two_step_var.get(),
            sharpen_enabled=self.sharpen_var.get(),
            unsharp_radius=float(self.radius_var.get()),
            unsharp_percent=int(float(self.percent_var.get())),
            unsharp_threshold=int(float(self.threshold_var.get())),
            pad_to_square=self.pad_square_var.get(),
            background_mode=self.background_mode_var.get(),
            custom_background_color=self.parse_rgb(self.custom_bg_color_var.get()),
            output_format_mode=self.output_format_var.get(),
            suffix=self.suffix_var.get(),
            recursive=self.recursive_var.get(),
            overwrite=self.overwrite_var.get(),
            quality=int(self.quality_var.get()),
        )

    def run_clicked(self):
        try:
            settings = self.build_settings()
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self.run_button.config(state="disabled")
        self.progress["value"] = 0
        self.status_label.config(text="Running...")
        self.log_text.delete("1.0", tk.END)

        def worker():
            try:
                def progress_callback(idx: int, total: int, message: str):
                    self.root.after(0, lambda: self.update_progress(idx, total, message))

                summary = process_batch(settings, progress_callback=progress_callback)
                self.root.after(0, lambda: self.finish_success(summary))
            except Exception as exc:
                self.root.after(0, lambda: self.finish_error(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def update_progress(self, idx: int, total: int, message: str):
        self.progress["maximum"] = max(total, 1)
        self.progress["value"] = idx
        self.status_label.config(text=f"Processing {idx} of {total}")
        self.log(message)

    def finish_success(self, summary: dict[str, int]):
        self.run_button.config(state="normal")
        self.status_label.config(text="Done")
        self.log("")
        self.log("Done.")
        self.log(f"Total files found: {summary['total']}")
        self.log(f"Saved: {summary['saved']}")
        self.log(f"Skipped existing: {summary['skipped']}")
        self.log(f"Failed: {summary['failed']}")
        self.log(f"Scale Percent: {summary.get('scale_percent', 'N/A')}%")
        self.log(f"Bleed per side: {summary['bleed_px']} px")
        self.log(f"Final output size: {summary['final_width_px']} x {summary['final_height_px']} px")
        messagebox.showinfo("Complete", "Processing completed.")

    def finish_error(self, error_message: str):
        self.run_button.config(state="normal")
        self.status_label.config(text="Error")
        self.log(f"[ERROR] {error_message}")
        messagebox.showerror("Processing failed", error_message)


def main():
    root = tk.Tk()
    app = UpscaleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
