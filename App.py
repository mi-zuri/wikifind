import tkinter as tk
from tkinter import ttk, messagebox
import threading
from Engine import Engine
from PIL import Image, ImageTk


# Issue: history storing logic should be kept in Engine
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Wikipedia Recommendation Engine")
        self.root.geometry("1400x800")

        self.engine = Engine()

        # Progress tracking
        self.scrape_progress = tk.DoubleVar(value=0)
        self.prep_progress = tk.DoubleVar(value=0)
        self.status_text = tk.StringVar(value="Data: None")
        self.scrape_detail_text = tk.StringVar(value="")
        self.prep_detail_text = tk.StringVar(value="")
        self.vectorize_status_text = tk.StringVar(value="")

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Create pages
        self.page1 = ttk.Frame(self.notebook)
        self.page2 = ttk.Frame(self.notebook)

        self.notebook.add(self.page1, text="Load & Process")
        self.notebook.add(self.page2, text="Recommendations")

        # Recommendation tracking
        self.visited_titles = []
        self.current_recommendations = []
        self.current_plot_path = None
        self._resize_timer = None

        self.build_page1()
        self.build_page2()

    def build_page1(self):
        container = ttk.Frame(self.page1)
        container.pack(fill='both', expand=True, padx=10, pady=10)

        data_frame = ttk.LabelFrame(container, text="Get Data", padding=10)
        data_frame.pack(fill='x', pady=(0, 15))

        top_row = ttk.Frame(data_frame)
        top_row.pack(fill='x', pady=(0, 10))

        ttk.Label(top_row, text="Seed Article:").pack(side='left', padx=(0, 5))
        self.seed_entry = ttk.Entry(top_row, width=30)
        self.seed_entry.insert(0, "Poland")
        self.seed_entry.pack(side='left', padx=(0, 20))

        ttk.Label(top_row, text="Links/Page:").pack(side='left', padx=(0, 5))
        self.links_entry = ttk.Entry(top_row, width=10)
        self.links_entry.insert(0, "10")
        self.links_entry.pack(side='left', padx=(0, 10))

        ttk.Label(top_row, text="Limit:").pack(side='left', padx=(0, 5))
        self.limit_entry = ttk.Entry(top_row, width=10)
        self.limit_entry.insert(0, "200")
        self.limit_entry.pack(side='left')

        button_row = ttk.Frame(data_frame)
        button_row.pack(fill='x', pady=(0, 10))

        self.scrape_btn = ttk.Button(button_row, text="Scrape",
                                     command=lambda: self.run_get_data(scrape=True))
        self.scrape_btn.pack(side='left', expand=True, fill='x', padx=(0, 5))

        self.load_btn = ttk.Button(button_row, text="Load",
                                   command=lambda: self.run_get_data(scrape=False))
        self.load_btn.pack(side='left', expand=True, fill='x', padx=(5, 0))

        ttk.Label(data_frame, text="Scraping Progress:").pack(anchor='w', pady=(5, 2))
        self.scrape_progressbar = ttk.Progressbar(data_frame, variable=self.scrape_progress,
                                                  maximum=100, mode='determinate')
        self.scrape_progressbar.pack(fill='x', pady=(0, 2))
        self.scrape_detail_label = ttk.Label(data_frame, textvariable=self.scrape_detail_text,
                                             font=('TkDefaultFont', 8))
        self.scrape_detail_label.pack(anchor='w', pady=(0, 10))

        ttk.Label(data_frame, text="Preprocessing Progress:").pack(anchor='w', pady=(5, 2))
        self.prep_progressbar = ttk.Progressbar(data_frame, variable=self.prep_progress,
                                                maximum=100, mode='determinate')
        self.prep_progressbar.pack(fill='x', pady=(0, 2))
        self.prep_detail_label = ttk.Label(data_frame, textvariable=self.prep_detail_text,
                                           font=('TkDefaultFont', 8))
        self.prep_detail_label.pack(anchor='w', pady=(0, 10))

        self.status_label = ttk.Label(data_frame, textvariable=self.status_text,
                                      font=('TkDefaultFont', 10, 'bold'))
        self.status_label.pack(anchor='w')

        vec_frame = ttk.LabelFrame(container, text="Vectorize", padding=10)
        vec_frame.pack(fill='both', expand=True)

        params_frame = ttk.Frame(vec_frame)
        params_frame.pack(fill='x', pady=(0, 10))

        row1 = ttk.Frame(params_frame)
        row1.pack(fill='x', pady=2)
        ttk.Label(row1, text="Max Features:", width=15).pack(side='left')
        self.max_features_entry = ttk.Entry(row1, width=12)
        self.max_features_entry.insert(0, "10000")
        self.max_features_entry.pack(side='left', padx=(0, 20))

        ttk.Label(row1, text="Max DF:", width=10).pack(side='left')
        self.max_df_entry = ttk.Entry(row1, width=12)
        self.max_df_entry.insert(0, "0.8")
        self.max_df_entry.pack(side='left')

        row2 = ttk.Frame(params_frame)
        row2.pack(fill='x', pady=2)
        ttk.Label(row2, text="Min DF:", width=15).pack(side='left')
        self.min_df_entry = ttk.Entry(row2, width=12)
        self.min_df_entry.insert(0, "3")
        self.min_df_entry.pack(side='left', padx=(0, 20))

        ttk.Label(row2, text="N-gram Max:", width=10).pack(side='left')
        self.ngram_entry = ttk.Entry(row2, width=12)
        self.ngram_entry.insert(0, "3")
        self.ngram_entry.pack(side='left')

        # LSA options
        lsa_frame = ttk.Frame(vec_frame)
        lsa_frame.pack(fill='x', pady=(10, 0))

        self.use_lsa_var = tk.BooleanVar(value=True)
        self.lsa_check = ttk.Checkbutton(lsa_frame, text="Use LSA",
                                         variable=self.use_lsa_var,
                                         command=self.toggle_lsa)
        self.lsa_check.pack(anchor='w')

        # Auto components frame
        self.auto_frame = ttk.Frame(vec_frame)
        self.auto_frame.pack(fill='x', pady=(5, 0))

        self.auto_components_var = tk.BooleanVar(value=True)
        self.auto_check = ttk.Checkbutton(self.auto_frame, text="Auto Components",
                                          variable=self.auto_components_var,
                                          command=self.toggle_auto_components)
        self.auto_check.pack(anchor='w')

        # Components/Variance row
        self.comp_frame = ttk.Frame(vec_frame)
        self.comp_frame.pack(fill='x', pady=(5, 10))

        # Target variance (shown when auto is ON)
        self.variance_frame = ttk.Frame(self.comp_frame)
        ttk.Label(self.variance_frame, text="Target Variance:", width=15).pack(side='left')
        self.target_variance_entry = ttk.Entry(self.variance_frame, width=12)
        self.target_variance_entry.insert(0, "0.40")
        self.target_variance_entry.pack(side='left')

        # N components (shown when auto is OFF)
        self.ncomp_frame = ttk.Frame(self.comp_frame)
        ttk.Label(self.ncomp_frame, text="N Components:", width=15).pack(side='left')
        self.n_components_entry = ttk.Entry(self.ncomp_frame, width=12)
        self.n_components_entry.insert(0, "100")
        self.n_components_entry.pack(side='left')

        self.toggle_auto_components()

        # Vectorize button
        self.vectorize_btn = ttk.Button(vec_frame, text="Vectorize",
                                        command=self.run_vectorize)
        self.vectorize_btn.pack(fill='x', pady=(10, 5))

        # Vectorization status indicator
        self.vectorize_status_label = ttk.Label(vec_frame, textvariable=self.vectorize_status_text,
                                                font=('TkDefaultFont', 9, 'italic'),
                                                foreground='white')
        self.vectorize_status_label.pack(anchor='center', pady=(0, 5))

    def toggle_lsa(self):
        if self.use_lsa_var.get():
            self.auto_check.state(['!disabled'])
            self.toggle_auto_components()
        else:
            self.auto_check.state(['disabled'])
            self.variance_frame.pack_forget()
            self.ncomp_frame.pack_forget()

    def toggle_auto_components(self):
        if not self.use_lsa_var.get():
            return

        if self.auto_components_var.get():
            self.ncomp_frame.pack_forget()
            self.variance_frame.pack(fill='x')
        else:
            self.variance_frame.pack_forget()
            self.ncomp_frame.pack(fill='x')

    def update_scrape_progress(self, current, total, title):
        percentage = (current / total * 100) if total > 0 else 0
        self.scrape_progress.set(percentage)
        self.scrape_detail_text.set(f"Scraped {current}/{total} articles | Current: {title}")
        self.status_text.set(f"Scraping")
        self.root.update_idletasks()

    def update_prep_progress(self, current, total):
        percentage = (current / total * 100) if total > 0 else 0
        self.prep_progress.set(percentage)
        self.prep_detail_text.set(f"Processed {current}/{total} articles")
        self.status_text.set(f"Preprocessing")
        self.root.update_idletasks()

    def run_get_data(self, scrape):
        self.clear_history()

        def task():
            try:
                # Reset progress
                self.scrape_progress.set(0)
                self.prep_progress.set(0)
                self.scrape_detail_text.set("")
                self.prep_detail_text.set("")

                # Disable buttons
                self.scrape_btn.state(['disabled'])
                self.load_btn.state(['disabled'])

                seed = self.seed_entry.get()
                links_per_page = int(self.links_entry.get())
                limit = int(self.limit_entry.get())

                if scrape:
                    self.status_text.set(f"Scraping...")
                else:
                    self.status_text.set(f"Loading...")

                self.engine.get_data(
                    scrape=scrape,
                    progress_scrape=self.update_scrape_progress,
                    progress_prep=self.update_prep_progress,
                    seed_article=seed,
                    links_per_page=links_per_page,
                    limit=limit
                )

                # Update status
                article_count = len(self.engine.raw_data) if hasattr(self.engine, 'raw_data') else 0
                self.status_text.set(f"Data: {seed} ({article_count} articles)")

                messagebox.showinfo("Success", "Data loaded successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to get data: {str(e)}")
                self.status_text.set("Data: Error")
            finally:
                # Re-enable buttons
                self.scrape_btn.state(['!disabled'])
                self.load_btn.state(['!disabled'])

        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def run_vectorize(self):
        self.clear_history()

        def task():
            try:
                # Disable button and show status
                self.vectorize_btn.state(['disabled'])
                self.vectorize_status_text.set("⏳ Vectorization in progress...")

                # Get parameters
                max_features = int(self.max_features_entry.get())
                max_df = float(self.max_df_entry.get())
                min_df = int(self.min_df_entry.get())
                ngram_max = int(self.ngram_entry.get())
                use_lsa = self.use_lsa_var.get()
                auto_components = self.auto_components_var.get()

                if use_lsa:
                    if auto_components:
                        target_variance = float(self.target_variance_entry.get())
                        n_components = 100  # Will be overridden by auto
                    else:
                        n_components = int(self.n_components_entry.get())
                        target_variance = 0.40
                else:
                    n_components = 100
                    target_variance = 0.40

                self.engine.vectorize(
                    max_features=max_features,
                    use_lsa=use_lsa,
                    max_df=max_df,
                    min_df=min_df,
                    ngram_max=ngram_max,
                    n_components=n_components,
                    auto_components=auto_components,
                    target_variance=target_variance
                )

                self.vectorize_status_text.set("✓ Vectorization completed successfully!")
                messagebox.showinfo("Success", "Vectorization completed!")

                # Switch to page 2
                self.notebook.select(self.page2)

            except Exception as e:
                self.vectorize_status_text.set("✗ Vectorization failed!")
                messagebox.showerror("Error", f"Vectorization failed: {str(e)}")
            finally:
                # Re-enable button
                self.vectorize_btn.state(['!disabled'])

        thread = threading.Thread(target=task, daemon=True)
        thread.start()

    def clear_history(self):
        # Reset lists and paths
        self.visited_titles = []
        self.current_recommendations = []
        self.current_plot_path = None

        # Clear UI elements
        self.history_listbox.delete(0, tk.END)
        self.rec_listbox.delete(0, tk.END)
        self.explain_text.delete('1.0', tk.END)

        self.coverage_text.config(state='normal')
        self.coverage_text.delete('1.0', tk.END)
        self.coverage_text.config(state='disabled')

        self.plot_canvas.delete('all')

    def build_page2(self):
        main_container = ttk.Frame(self.page2)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)

        main_container.columnconfigure(0, weight=5, uniform='group1')
        main_container.columnconfigure(1, weight=9, uniform='group1')
        main_container.columnconfigure(2, weight=5, uniform='group1')
        main_container.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(main_container)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5))

        middle_frame = ttk.Frame(main_container)
        middle_frame.grid(row=0, column=1, sticky='nsew', padx=5)

        right_frame = ttk.Frame(main_container)
        right_frame.grid(row=0, column=2, sticky='nsew', padx=(5, 0))

        left_frame.columnconfigure(0, weight=1)

        left_frame.rowconfigure(0, weight=1, uniform='left_split')
        left_frame.rowconfigure(1, weight=1, uniform='left_split')

        select_frame = ttk.LabelFrame(left_frame, text="Select Article", padding=5)
        select_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 5))

        ttk.Label(select_frame, text="Search:").pack(anchor='w')
        self.article_search_var = tk.StringVar()
        self.article_search_var.trace_add('write', self.filter_articles)
        search_entry = ttk.Entry(select_frame, textvariable=self.article_search_var)
        search_entry.pack(fill='x', pady=(2, 5))

        list_frame = ttk.Frame(select_frame)
        list_frame.pack(fill='both', expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side='right', fill='y')

        self.article_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.article_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.article_listbox.yview)

        self.add_history_btn = ttk.Button(select_frame, text="Add to History",
                                          command=self.add_to_history)
        self.add_history_btn.pack(fill='x', pady=(5, 0))

        history_frame = ttk.LabelFrame(left_frame, text="Visit History", padding=5)
        history_frame.grid(row=1, column=0, sticky='nsew', pady=(5, 0))

        hist_list_frame = ttk.Frame(history_frame)
        hist_list_frame.pack(fill='both', expand=True)

        hist_scrollbar = ttk.Scrollbar(hist_list_frame)
        hist_scrollbar.pack(side='right', fill='y')

        self.history_listbox = tk.Listbox(hist_list_frame, yscrollcommand=hist_scrollbar.set)
        self.history_listbox.pack(side='left', fill='both', expand=True)
        hist_scrollbar.config(command=self.history_listbox.yview)

        self.clear_hist_btn = ttk.Button(history_frame, text="Clear History",
                                         command=self.clear_history)
        self.clear_hist_btn.pack(fill='x', pady=(5, 0))

        middle_frame.columnconfigure(0, weight=1)

        middle_frame.rowconfigure(0, weight=1, uniform='mid_split')
        middle_frame.rowconfigure(1, weight=2, uniform='mid_split')

        rec_frame = ttk.LabelFrame(middle_frame, text="Recommendations", padding=5)
        rec_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 5))

        rec_list_frame = ttk.Frame(rec_frame)
        rec_list_frame.pack(fill='both', expand=True)

        rec_scrollbar = ttk.Scrollbar(rec_list_frame)
        rec_scrollbar.pack(side='right', fill='y')

        self.rec_listbox = tk.Listbox(rec_list_frame, yscrollcommand=rec_scrollbar.set)
        self.rec_listbox.pack(side='left', fill='both', expand=True)
        rec_scrollbar.config(command=self.rec_listbox.yview)
        self.rec_listbox.bind('<<ListboxSelect>>', self.on_recommendation_select)

        plot_frame = ttk.LabelFrame(middle_frame, text="Vector Space Visualization", padding=5)
        plot_frame.grid(row=1, column=0, sticky='nsew', pady=(5, 0))

        self.plot_canvas = tk.Canvas(plot_frame, bg='gray12')
        self.plot_canvas.pack(fill='both', expand=True)

        self.plot_canvas.bind('<Configure>', lambda e: self.resize_plot())

        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=2, uniform='right_split')
        right_frame.rowconfigure(1, weight=1, uniform='right_split')

        explain_frame = ttk.LabelFrame(right_frame, text="Similarity Explanation", padding=5)
        explain_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 5))

        explain_text_frame = ttk.Frame(explain_frame)
        explain_text_frame.pack(fill='both', expand=True)

        explain_scrollbar = ttk.Scrollbar(explain_text_frame)
        explain_scrollbar.pack(side='right', fill='y')

        self.explain_text = tk.Text(explain_text_frame, wrap='word',
                                    yscrollcommand=explain_scrollbar.set)
        self.explain_text.pack(side='left', fill='both', expand=True)
        explain_scrollbar.config(command=self.explain_text.yview)

        coverage_frame = ttk.LabelFrame(right_frame, text="Recommendation Coverage", padding=5)
        coverage_frame.grid(row=1, column=0, sticky='nsew', pady=(5, 0))

        self.coverage_text = tk.Text(coverage_frame, height=6, wrap='word', state='disabled')
        self.coverage_text.pack(fill='both', expand=True)

    def filter_articles(self, *args):
        if not hasattr(self.engine, 'raw_data'):
            return

        search_text = self.article_search_var.get().lower()
        self.article_listbox.delete(0, tk.END)

        for title in self.engine.raw_data['title']:
            if search_text in title.lower():
                self.article_listbox.insert(tk.END, title)

    def add_to_history(self):
        selection = self.article_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an article first")
            return

        title = self.article_listbox.get(selection[0])

        if title in self.visited_titles:
            self.visited_titles.remove(title)
        self.visited_titles.append(title)

        self.update_history_display()

        self.update_recommendations()

        self.explain_text.delete('1.0', tk.END)

    def update_history_display(self):
        self.history_listbox.delete(0, tk.END)
        for title in reversed(self.visited_titles):
            self.history_listbox.insert(tk.END, title)

    def update_recommendations(self):
        if not self.visited_titles:
            return

        try:
            self.current_recommendations = self.engine.get_recommendations(
                self.visited_titles, top_n=20
            )

            self.rec_listbox.delete(0, tk.END)
            for i, (title, sim, idx) in enumerate(self.current_recommendations, 1):
                self.rec_listbox.insert(tk.END, f"{i}. {title}")

            coverage = self.engine.calculate_recommendation_coverage(
                self.visited_titles, self.current_recommendations
            )

            categories = coverage['shared_categories']
            categories_text = '\n'.join(f"- {cat}" for cat in categories)

            self.coverage_text.config(state='normal')
            self.coverage_text.delete('1.0', tk.END)
            self.coverage_text.insert(
                '1.0',
                f"Match Precision: {coverage['precision_percent']:.1f}% "
                f"({coverage['match_count']}/{coverage['total_recommendations']})\n"
                f"\nShared Categories:\n"
                f"{categories_text}"
            )
            self.coverage_text.config(state='disabled')

            self.update_vector_plot()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to get recommendations: {str(e)}")

    def on_recommendation_select(self, event):
        selection = self.rec_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        if idx >= len(self.current_recommendations):
            return

        title, sim, rec_idx = self.current_recommendations[idx]

        try:
            explanation = self.engine.explain_similarity(
                self.visited_titles, rec_idx, top_n=5
            )

            self.explain_text.delete('1.0', tk.END)
            self.explain_text.insert('1.0', f"Explanation for: {title}\n")
            self.explain_text.insert(tk.END, f"Similarity: {sim:.3f}\n\n")
            self.explain_text.insert(tk.END, "Top Contributing Factors:\n")
            self.explain_text.insert(tk.END, "-" * 40 + "\n")

            for i, (feature, score) in enumerate(explanation, 1):
                self.explain_text.insert(tk.END, f"{i}. {feature}\n")
                self.explain_text.insert(tk.END, f"   Score: {score:.4f}\n\n")

        except Exception as e:
            self.explain_text.delete('1.0', tk.END)
            self.explain_text.insert('1.0', f"Error getting explanation: {str(e)}")

    def resize_plot(self):
        if hasattr(self, 'current_plot_path') and self.current_plot_path:
            self.display_plot(self.current_plot_path)

    def display_plot(self, image_path):
        try:
            img = Image.open(image_path)

            canvas_width = self.plot_canvas.winfo_width()
            canvas_height = self.plot_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                # Calculate scaling to fit canvas while maintaining aspect ratio
                img_ratio = img.width / img.height
                canvas_ratio = canvas_width / canvas_height

                if img_ratio > canvas_ratio:
                    # Image is wider than canvas
                    new_width = canvas_width
                    new_height = int(canvas_width / img_ratio)
                else:
                    # Image is taller than canvas
                    new_height = canvas_height
                    new_width = int(canvas_height * img_ratio)

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)

            self.plot_canvas.delete('all')
            x = (canvas_width - img.width) // 2 if canvas_width > 1 else 0
            y = (canvas_height - img.height) // 2 if canvas_height > 1 else 0
            self.plot_canvas.create_image(x, y, anchor='nw', image=photo)
            self.plot_canvas.image = photo  # Keep reference

        except Exception as e:
            self.plot_canvas.delete('all')
            self.plot_canvas.create_text(
                self.plot_canvas.winfo_width() // 2,
                self.plot_canvas.winfo_height() // 2,
                text=f"Error loading plot: {str(e)}",
                fill='white'
            )

    def update_vector_plot(self):
        try:
            plot_path = "plots/vector_space_map.png"
            self.engine.plot_vector_space(
                self.visited_titles,
                self.current_recommendations,
                filename=plot_path
            )

            self.current_plot_path = plot_path
            self.display_plot(plot_path)

        except Exception as e:
            self.plot_canvas.delete('all')
            self.plot_canvas.create_text(
                self.plot_canvas.winfo_width() // 2,
                self.plot_canvas.winfo_height() // 2,
                text=f"Error generating plot: {str(e)}",
                fill='white'
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

# Example usage:
# Load: Cyberpunk_2077
# Vectorize with 20k features, 0.7 Max DF, 10 Min DF, 2 N-gram and 500 components