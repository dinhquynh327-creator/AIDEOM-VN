import nbformat as nbf
import os

def create_notebook(filename, title, markdown_desc, code_blocks):
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Title and description
    cells.append(nbf.v4.new_markdown_cell(f"# {title}\n{markdown_desc}"))
    
    # Add common imports
    imports = "import sys\nimport os\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
    imports += "sys.path.append(os.path.abspath('..'))\n"
    imports += "from src.data_loader import get_data"
    cells.append(nbf.v4.new_code_cell(imports))
    
    for i, code in enumerate(code_blocks):
        cells.append(nbf.v4.new_markdown_cell(f"### Yêu cầu {i+1}"))
        cells.append(nbf.v4.new_code_cell(code))
        
    nb['cells'] = cells
    
    with open(f"notebooks/{filename}", 'w') as f:
        nbf.write(nb, f)

# --- BÀI 1 ---
b1_c1 = """
data = get_data()
years = data.macro_years
Y = data.macro_Y
K = data.macro_K
L = data.macro_L
D = data.macro_D
AI = data.macro_AI
H = data.macro_H

alpha, beta, gamma, delta, theta = 0.33, 0.42, 0.10, 0.08, 0.07
s = alpha + beta + gamma + delta + theta
a, b, g, d, t = alpha/s, beta/s, gamma/s, delta/s, theta/s

A = Y / (K**a * L**b * D**g * AI**d * H**t)
print("TFP ước lượng (A_t):", A)

plt.figure(figsize=(6,4))
plt.plot(years, A, marker='o')
plt.title('Xu hướng TFP (A_t) 2020-2025')
plt.grid()
plt.show()
"""

b1_c2 = """
A_avg = A.mean()
Y_hat = A_avg * (K**a * L**b * D**g * AI**d * H**t)
MAPE = np.mean(np.abs((Y - Y_hat)/Y)) * 100
print(f"Chỉ số MAPE: {MAPE:.2f}%")
"""

b1_c3 = """
contrib_K = a * np.mean(np.diff(np.log(K)))
contrib_L = b * np.mean(np.diff(np.log(L)))
contrib_D = g * np.mean(np.diff(np.log(D)))
contrib_AI = d * np.mean(np.diff(np.log(AI)))
contrib_H = t * np.mean(np.diff(np.log(H)))
contrib_TFP = np.mean(np.diff(np.log(A)))

total = contrib_K + contrib_L + contrib_D + contrib_AI + contrib_H + contrib_TFP
labels = ['K', 'L', 'D', 'AI', 'H', 'TFP']
values = [contrib_K, contrib_L, contrib_D, contrib_AI, contrib_H, contrib_TFP]
pct = [v/total * 100 for v in values]

plt.bar(labels, pct)
plt.title('Phân rã tăng trưởng GDP 2020-2025 (%)')
plt.show()
"""

b1_c4 = """
K_2030 = K[-1] * (1.06)**5
L_2030 = L[-1] * (1.06)**5
D_2030 = 30
AI_2030 = 100
H_2030 = 35
A_2030 = A[-1] * (1.012)**5

Y_2030 = A_2030 * K_2030**a * L_2030**b * D_2030**g * AI_2030**d * H_2030**t
print(f"Dự báo GDP 2030: {Y_2030:.2f} nghìn tỷ VND")
"""
create_notebook('bai01_notebook.ipynb', 'Bài 1: Hàm sản xuất Cobb-Douglas', 'Phân tích TFP và tăng trưởng.', [b1_c1, b1_c2, b1_c3, b1_c4])

# --- BÀI 2 ---
b2_c1 = """
import pulp
m = pulp.LpProblem('Resource_Allocation', pulp.LpMaximize)
# x1: I, x2: D, x3: AI, x4: H
x1 = pulp.LpVariable('x1', lowBound=25)
x2 = pulp.LpVariable('x2', lowBound=0)
x3 = pulp.LpVariable('x3', lowBound=15)
x4 = pulp.LpVariable('x4', lowBound=0)

m += 0.85*x1 + 1.20*x2 + 0.95*x3 + 1.35*x4
m += x1 + x2 + x3 + x4 <= 100
m += x2 + x4 >= 0.35*(x1 + x2 + x3 + x4)

m.solve(pulp.PULP_CBC_CMD(msg=False))
print("Z* =", pulp.value(m.objective))
for v in m.variables():
    print(v.name, "=", v.varValue)
"""
create_notebook('bai02_notebook.ipynb', 'Bài 2: Quy hoạch tuyến tính Ngân sách', 'Phân bổ ngân sách cơ bản.', [b2_c1])

# --- BÀI 4 ---
b4_c1 = """
import pulp
regions = ['NMM', 'RRD', 'NCC', 'CH', 'SE', 'MD']
items = ['I', 'D', 'AI', 'H']
beta = {
    ('NMM','I'):1.15, ('NMM','D'):0.85, ('NMM','AI'):0.55, ('NMM','H'):1.30,
    ('RRD','I'):0.95, ('RRD','D'):1.25, ('RRD','AI'):1.40, ('RRD','H'):1.05,
    ('NCC','I'):1.05, ('NCC','D'):0.95, ('NCC','AI'):0.85, ('NCC','H'):1.15,
    ('CH','I') :1.20, ('CH','D') :0.75, ('CH','AI') :0.45, ('CH','H') :1.35,
    ('SE','I') :0.90, ('SE','D') :1.30, ('SE','AI') :1.55, ('SE','H') :1.00,
    ('MD','I') :1.10, ('MD','D') :0.85, ('MD','AI') :0.65, ('MD','H') :1.25
}

m = pulp.LpProblem('LP_Regions', pulp.LpMaximize)
x = pulp.LpVariable.dicts('x', (regions, items), lowBound=0)

m += pulp.lpSum(beta[r,j]*x[r][j] for r in regions for j in items)
m += pulp.lpSum(x[r][j] for r in regions for j in items) <= 50000

for r in regions:
    m += pulp.lpSum(x[r][j] for j in items) >= 5000
    m += pulp.lpSum(x[r][j] for j in items) <= 12000
    
m += pulp.lpSum(x[r]['H'] for r in regions) >= 12000

m.solve(pulp.PULP_CBC_CMD(msg=False))
print("Z* =", pulp.value(m.objective))
"""
create_notebook('bai04_notebook.ipynb', 'Bài 4: Phân bổ ngân sách vùng', 'Dùng LP cho phân bổ đa vùng.', [b4_c1])

print("Tạo xong các notebook đại diện!")
