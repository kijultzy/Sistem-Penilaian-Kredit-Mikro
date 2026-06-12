import json

notebook_path = "TUBES DKA.ipynb"

# Load notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Loop through cells and apply modifications
for cell in nb["cells"]:
    # 1. Update Markdown header for Lapis 2
    if cell["cell_type"] == "markdown":
        for i, line in enumerate(cell["source"]):
            if "Lapis 2 - Logika Fuzzy From Scratch (22 Aturan)" in line:
                cell["source"][i] = line.replace("22 Aturan", "34 Aturan")
    
    # 2. Update code cells
    if cell["cell_type"] == "code":
        # Identify SEL 4: KEANGGOTAAN DAN INFERENSI FUZZY
        is_sel_4 = any("SEL 4: KEANGGOTAAN DAN INFERENSI FUZZY" in line for line in cell["source"])
        if is_sel_4:
            # Reconstruct the entire source with 34 rules
            new_source = [
                "# ==============================================================================\n",
                "# SEL 4: KEANGGOTAAN DAN INFERENSI FUZZY (34 ATURAN)\n",
                "# ==============================================================================\n",
                "def segitiga(x, a, b, c):\n",
                "    if x <= a or x >= c: return 0.0\n",
                "    if a < x <= b: return (x - a) / (b - a) if b != a else 1.0\n",
                "    return (c - x) / (c - b) if c != b else 1.0\n",
                "\n",
                "def trapesium(x, a, b, c, d):\n",
                "    # Menggunakan perbandingan strictly kurang/lebih untuk mengakomodasi boundary 0\n",
                "    if x < a or x > d: return 0.0\n",
                "    if a < x < b: return (x - a) / (b - a) if b != a else 1.0\n",
                "    if b <= x <= c: return 1.0\n",
                "    return (d - x) / (d - c) if d != c else 1.0\n",
                "\n",
                "def get_fuzzy_values(row):\n",
                "    inc = row['Annual_Income']\n",
                "    debt = row['Outstanding_Debt']\n",
                "    ir = row['Interest_Rate']\n",
                "    del_day = row['Delay_from_due_date']\n",
                "    num_del = row['Num_of_Delayed_Payment']\n",
                "    \n",
                "    return {\n",
                "        'inc': {\n",
                "            'Rendah': trapesium(inc, 0, 0, 30000, 50000),\n",
                "            'Sedang': segitiga(inc, 30000, 65000, 100000),\n",
                "            'Tinggi': trapesium(inc, 80000, 100000, 1500000, 2000000)\n",
                "        },\n",
                "        'debt': {\n",
                "            'Sedikit': trapesium(debt, 0, 0, 1000, 2000),\n",
                "            'Sedang':  segitiga(debt, 1000, 2500, 4000),\n",
                "            'Banyak':  trapesium(debt, 3000, 5000, 100000, 100000)\n",
                "        },\n",
                "        'ir': {\n",
                "            'Rendah': trapesium(ir, 0, 0, 8, 12),\n",
                "            'Sedang': segitiga(ir, 8, 15, 22),\n",
                "            'Tinggi': trapesium(ir, 18, 25, 35, 100)\n",
                "        },\n",
                "        'del': {\n",
                "            'Singkat': trapesium(del_day, 0, 0, 10, 20),\n",
                "            'Sedang':  segitiga(del_day, 10, 30, 45),\n",
                "            'Lama':    trapesium(del_day, 35, 60, 200, 200)\n",
                "        },\n",
                "        'num': {\n",
                "            'Jarang':   trapesium(num_del, 0, 0, 5, 8),\n",
                "            'Sering':   segitiga(num_del, 5, 12, 18),\n",
                "            'SgtSering':trapesium(num_del, 15, 25, 100, 100)\n",
                "        }\n",
                "    }\n",
                "\n",
                "def evaluate_34_rules(fz):\n",
                "    # --- OUTPUT: BAIK ---\n",
                "    r01 = min(fz['inc']['Tinggi'], fz['debt']['Sedikit'], fz['ir']['Rendah'], fz['del']['Singkat'], fz['num']['Jarang'])\n",
                "    r02 = min(fz['inc']['Tinggi'], fz['debt']['Sedang'],  fz['ir']['Sedang'], fz['del']['Singkat'], fz['num']['Jarang'])\n",
                "    r03 = min(fz['inc']['Sedang'], fz['debt']['Sedikit'], fz['ir']['Rendah'], fz['del']['Singkat'], fz['num']['Jarang'])\n",
                "    r04 = min(fz['inc']['Tinggi'], fz['debt']['Sedikit'], fz['ir']['Tinggi'], fz['del']['Sedang'],  fz['num']['Jarang'])\n",
                "    r16 = min(fz['inc']['Sedang'], fz['debt']['Sedikit'], fz['ir']['Sedang'], fz['del']['Singkat'], fz['num']['Jarang'])\n",
                "    r17 = min(fz['inc']['Tinggi'], fz['debt']['Sedang'],  fz['ir']['Rendah'], fz['del']['Singkat'], fz['num']['Jarang'])\n",
                "    r23 = min(fz['inc']['Tinggi'], fz['debt']['Sedikit'], fz['ir']['Sedang'], fz['del']['Singkat'], fz['num']['Jarang'])\n",
                "    r24 = min(fz['inc']['Sedang'], fz['debt']['Sedang'],  fz['ir']['Rendah'], fz['del']['Singkat'], fz['num']['Jarang'])\n",
                "    \n",
                "    # --- OUTPUT: STANDAR ---\n",
                "    r05 = min(fz['inc']['Sedang'], fz['debt']['Sedang'],  fz['ir']['Sedang'], fz['del']['Sedang'],  fz['num']['Sering'])\n",
                "    r06 = min(fz['inc']['Rendah'], fz['debt']['Sedikit'], fz['ir']['Rendah'], fz['del']['Singkat'], fz['num']['Jarang'])\n",
                "    r07 = min(fz['inc']['Tinggi'], fz['debt']['Banyak'],  fz['ir']['Sedang'], fz['del']['Singkat'], fz['num']['Jarang'])\n",
                "    r08 = min(fz['inc']['Sedang'], fz['debt']['Banyak'],  fz['ir']['Tinggi'], fz['del']['Singkat'], fz['num']['Jarang'])\n",
                "    r09 = min(fz['inc']['Rendah'], fz['debt']['Sedang'],  fz['ir']['Rendah'], fz['del']['Singkat'], fz['num']['Jarang'])\n",
                "    r10 = min(fz['inc']['Tinggi'], fz['debt']['Sedang'],  fz['ir']['Tinggi'], fz['del']['Sedang'],  fz['num']['Sering'])\n",
                "    r18 = min(fz['inc']['Sedang'], fz['debt']['Sedang'],  fz['ir']['Sedang'], fz['del']['Singkat'], fz['num']['Jarang'])\n",
                "    r19 = min(fz['inc']['Rendah'], fz['debt']['Sedikit'], fz['ir']['Sedang'], fz['del']['Singkat'], fz['num']['Jarang'])\n",
                "    r20 = min(fz['inc']['Sedang'], fz['debt']['Sedang'],  fz['ir']['Tinggi'], fz['del']['Singkat'], fz['num']['Jarang'])\n",
                "    r33 = min(fz['inc']['Sedang'], fz['debt']['Sedikit'], fz['ir']['Sedang'], fz['del']['Singkat'], fz['num']['Sering'])\n",
                "    r34 = min(fz['inc']['Tinggi'], fz['debt']['Sedikit'], fz['ir']['Sedang'], fz['del']['Singkat'], fz['num']['Sering'])\n",
                "    \n",
                "    # --- OUTPUT: BURUK ---\n",
                "    r11 = min(fz['inc']['Rendah'], fz['debt']['Banyak'],  fz['ir']['Tinggi'], fz['del']['Lama'],    fz['num']['SgtSering'])\n",
                "    r12 = min(fz['inc']['Tinggi'], fz['debt']['Banyak'],  fz['ir']['Tinggi'], fz['del']['Lama'],    fz['num']['SgtSering'])\n",
                "    r13 = min(fz['inc']['Sedang'], fz['debt']['Banyak'],  fz['ir']['Sedang'], fz['del']['Sedang'],  fz['num']['Sering'])\n",
                "    r14 = min(fz['inc']['Rendah'], fz['debt']['Sedang'],  fz['ir']['Sedang'], fz['del']['Lama'],    fz['num']['Sering'])\n",
                "    r15 = min(fz['inc']['Sedang'], fz['debt']['Sedikit'], fz['ir']['Tinggi'], fz['del']['Lama'],    fz['num']['SgtSering'])\n",
                "    r21 = min(fz['inc']['Rendah'], fz['debt']['Sedang'],  fz['ir']['Tinggi'], fz['del']['Sedang'],  fz['num']['Sering'])\n",
                "    r22 = min(fz['inc']['Sedang'], fz['debt']['Banyak'],  fz['ir']['Tinggi'], fz['del']['Lama'],    fz['num']['Sering'])\n",
                "    r25 = min(fz['inc']['Rendah'], fz['debt']['Sedang'],  fz['ir']['Tinggi'], fz['del']['Lama'],    fz['num']['Sering'])\n",
                "    r26 = min(fz['inc']['Sedang'], fz['debt']['Sedang'],  fz['ir']['Tinggi'], fz['del']['Sedang'],  fz['num']['SgtSering'])\n",
                "    r27 = min(fz['inc']['Sedang'], fz['debt']['Sedang'],  fz['ir']['Tinggi'], fz['del']['Lama'],    fz['num']['SgtSering'])\n",
                "    r28 = min(fz['inc']['Sedang'], fz['debt']['Sedang'],  fz['ir']['Tinggi'], fz['del']['Sedang'],  fz['num']['Sering'])\n",
                "    r29 = min(fz['inc']['Rendah'], fz['debt']['Sedang'],  fz['ir']['Tinggi'], fz['del']['Lama'],    fz['num']['Jarang'])\n",
                "    r30 = min(fz['inc']['Rendah'], fz['debt']['Sedang'],  fz['ir']['Tinggi'], fz['del']['Singkat'], fz['num']['Sering'])\n",
                "    r31 = min(fz['inc']['Rendah'], fz['debt']['Sedikit'], fz['ir']['Tinggi'], fz['del']['Lama'],    fz['num']['Sering'])\n",
                "    r32 = min(fz['inc']['Sedang'], fz['debt']['Sedang'],  fz['ir']['Sedang'], fz['del']['Lama'],    fz['num']['SgtSering'])\n",
                "    \n",
                "    baik = max(r01, r02, r03, r04, r16, r17, r23, r24)\n",
                "    standar = max(r05, r06, r07, r08, r09, r10, r18, r19, r20, r33, r34)\n",
                "    buruk = max(r11, r12, r13, r14, r15, r21, r22, r25, r26, r27, r28, r29, r30, r31, r32)\n",
                "    \n",
                "    return baik, standar, buruk\n",
                "\n",
                "print(\"-> Sukses! Fungsi logika fuzzy dan 34 basis aturan berhasil didefinisikan.\")"
            ]
            cell["source"] = new_source
        
        # Identify SEL 6: EKSEKUSI BATCH & EVALUASI METRIK
        is_sel_6 = any("SEL 6: EKSEKUSI BATCH & EVALUASI METRIK" in line for line in cell["source"])
        if is_sel_6:
            for j, line in enumerate(cell["source"]):
                # Update evaluate_22_rules function call
                if "evaluate_22_rules" in line:
                    cell["source"][j] = line.replace("evaluate_22_rules", "evaluate_34_rules")

# Save updated notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[SUCCESS] TUBES DKA.ipynb updated successfully to 34 rules!")
