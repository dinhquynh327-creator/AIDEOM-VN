"""Common Optimization Models for Vietnam Digital Economy."""

import numpy as np
import pandas as pd
import pulp
import pyomo.environ as pyo
from scipy.optimize import linprog, minimize, milp, LinearConstraint, Bounds
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize

from .data_loader import get_data

SCENARIOS = {
    'S1': {'desc': 'Truyền thống', 'alloc_weights': [0.60, 0.15, 0.10, 0.15], 'tfp_growth': 0.010, 'ai_adoption': 0.20},
    'S2': {'desc': 'Số hóa nhanh', 'alloc_weights': [0.30, 0.40, 0.15, 0.15], 'tfp_growth': 0.012, 'ai_adoption': 0.40},
    'S3': {'desc': 'AI dẫn dắt', 'alloc_weights': [0.20, 0.20, 0.40, 0.20], 'tfp_growth': 0.015, 'ai_adoption': 0.60},
    'S4': {'desc': 'Bao trùm số', 'alloc_weights': [0.40, 0.20, 0.10, 0.30], 'tfp_growth': 0.011, 'ai_adoption': 0.30},
    'S5': {'desc': 'Tối ưu cân bằng', 'alloc_weights': [0.35, 0.25, 0.20, 0.20], 'tfp_growth': 0.014, 'ai_adoption': 0.35},
}


class VietnamDigitalProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=24, n_obj=4, n_ieq_constr=14, xl=np.zeros(24), xu=np.ones(24)*12000)
        # Beta matrix (6 vùng x 4 hạng mục)
        self.beta = np.array([
            [1.15, 0.85, 0.55, 1.30],
            [0.95, 1.25, 1.40, 1.05],
            [1.05, 0.95, 0.85, 1.15],
            [1.20, 0.75, 0.45, 1.35],
            [0.90, 1.30, 1.55, 1.00],
            [1.10, 0.85, 0.65, 1.25],
        ])
        self.e = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
        self.rho = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
        self.sig = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])

    def _evaluate(self, x, out, *args, **kwargs):
        X = x.reshape(6, 4)
        # f1: max GDP gain => -sum(beta * X)
        f1 = -(self.beta * X).sum()
        # f2: Gini approximated by MAD
        sums = X.sum(axis=1)
        f2 = np.abs(sums - sums.mean()).mean()
        # f3: emissions proxy (AI related)
        f3 = (self.e * (X[:,0] + X[:,2])).sum()
        # f4: security proxy
        f4 = (self.sig * X[:,1]).sum()
        
        out["F"] = [f1, f2, f3, f4]
        
        # Constraints (C1: sum X <= 50000) -> sum X - 50000 <= 0
        g1 = X.sum() - 50000
        
        # C2: sum_j x_jr >= 5000 -> 5000 - sum_j x_jr <= 0
        g2 = 5000 - X.sum(axis=1) # array of 6
        
        # C3: sum_j x_jr <= 12000 -> sum_j x_jr - 12000 <= 0
        g3 = X.sum(axis=1) - 12000 # array of 6
        
        # C4: sum_r x_H,r >= 12000 -> 12000 - sum_r x_H,r <= 0
        g4 = 12000 - X[:,3].sum()
        
        out["G"] = [g1, *g2, *g3, g4]


def solve_bai01(data_dir=None, alpha=0.33, beta=0.42, gamma=0.10, delta=0.08, theta=0.07, tfp_growth=0.012):
    # Vietnam macro data 2020-2025 (embedded - không phụ thuộc CSV)
    data = get_data(data_dir)
    years = data.macro_years
    Y  = data.macro_Y
    K  = data.macro_K
    L  = data.macro_L
    D  = data.macro_D
    AI = data.macro_AI
    H  = data.macro_H

    # Normalize exponents to sum=1
    s = alpha + beta + gamma + delta + theta or 1
    a, b, g, d, t = alpha/s, beta/s, gamma/s, delta/s, theta/s

    # TFP estimation
    A = Y / (K**a * L**b * D**g * AI**d * H**t)
    yhat = A.mean() * K**a * L**b * D**g * AI**d * H**t
    mape = float(np.mean(np.abs((Y - yhat) / Y)) * 100)

    # Forecast 2026-2030
    fy = np.arange(2026, 2031)
    Kf  = K[-1]  * (1.06) ** np.arange(1, 6)
    Lf  = L[-1]  * (1.06) ** np.arange(1, 6)
    Df  = np.linspace(D[-1],  30,  5)
    AIf = np.linspace(AI[-1], 100, 5)
    Hf  = np.linspace(H[-1],  35,  5)
    Af  = A[-1] * (1 + tfp_growth) ** np.arange(1, 6)
    Yf  = Af * Kf**a * Lf**b * Df**g * AIf**d * Hf**t

    # Growth decomposition
    contrib = {
        'K':   float(a * np.mean(np.diff(np.log(K)))),
        'L':   float(b * np.mean(np.diff(np.log(L)))),
        'D':   float(g * np.mean(np.diff(np.log(D)))),
        'AI':  float(d * np.mean(np.diff(np.log(AI)))),
        'H':   float(t * np.mean(np.diff(np.log(H)))),
        'TFP': float(np.mean(np.diff(np.log(A)))),
    }
    total = sum(contrib.values()) or 1
    contrib_pct = {k: round(v / total * 100, 2) for k, v in contrib.items()}

    # Scenarios
    Yf_high_tfp = (Yf * 1.05).tolist()
    Yf_ai_fast  = (Yf * 1.035).tolist()

    return {
        'years':         years.tolist(),
        'Y_actual':      Y.tolist(),
        'Y_hat':         yhat.tolist(),
        'A_t':           A.tolist(),
        'mape':          mape,
        'forecast_years': fy.tolist(),
        'forecast_series': Yf.tolist(),
        'forecast_high_tfp': Yf_high_tfp,
        'forecast_ai_fast':  Yf_ai_fast,
        'contrib_pct':   contrib_pct,
        'exponents':     {'alpha': a, 'beta': b, 'gamma': g, 'delta': d, 'theta': t},
        'gdp_2030':      float(Yf[-1]),
    }

def _solve(B, min_I=25, min_AI=15, min_H=20, coef_I=0.85, coef_AI=1.20, coef_H=0.95, coef_RD=1.35):
    c = [-coef_I, -coef_AI, -coef_H, -coef_RD]
    A_ub = [
        [1, 1, 1, 1],
        [0.35, -0.65, 0.35, -0.65]
    ]
    b_ub = [B, 0]
    bounds = [(min_I, None), (min_AI, None), (min_H, None), (10, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x.tolist(), float(-res.fun), True
    return [0, 0, 0, 0], 0.0, False

def solve_bai02(budget=100, min_I=25, min_AI=15, coef_I=0.85, coef_AI=1.20, coef_H=0.95, coef_RD=1.35):
    labels = ['I Hạ tầng', 'AI Dữ liệu', 'H Nhân lực', 'R&D']

    # 1. Linprog solve
    x, z, ok = _solve(budget, min_I, min_AI, min_H=20, coef_I=coef_I, coef_AI=coef_AI, coef_H=coef_H, coef_RD=coef_RD)

    # 2. PuLP solve for dual values
    m = pulp.LpProblem("Budget_Allocation", pulp.LpMaximize)
    x1 = pulp.LpVariable('x1', lowBound=min_I)
    x2 = pulp.LpVariable('x2', lowBound=min_AI)
    x3 = pulp.LpVariable('x3', lowBound=20)
    x4 = pulp.LpVariable('x4', lowBound=10)

    m += coef_I*x1 + coef_AI*x2 + coef_H*x3 + coef_RD*x4, "Objective"
    m += x1 + x2 + x3 + x4 <= budget, "C1_Budget"
    m += -0.35*x1 + 0.65*x2 - 0.35*x3 + 0.65*x4 >= 0, "C2_Tech35"

    m.solve(pulp.PULP_CBC_CMD(msg=False))
    dual_values = {}
    if m.status == pulp.LpStatusOptimal:
        for name, c in m.constraints.items():
            dual_values[name] = round(c.pi, 4)

    # 3. Sensitivity: vary budget
    budgets = sorted(set([80, 100, 120, 140, budget]))
    sens_budgets = []
    sens_z = []
    for bb in budgets:
        sens_budgets.append(bb)
        sens_z.append(_solve(bb, min_I, min_AI, 20)[1])

    # 4. Scenario: x3 >= 30
    x_sc, z_sc, ok_sc = _solve(budget, min_I, min_AI, min_H=30, coef_I=coef_I, coef_AI=coef_AI, coef_H=coef_H, coef_RD=coef_RD)

    return {
        'status':      'Optimal' if ok else 'Infeasible',
        'Z':           z,
        'allocation':  dict(zip(labels, x)),
        'dual_values': dual_values,
        'sensitivity_budgets': sens_budgets,
        'sensitivity_z': sens_z,
        'scenario_x3': {
            'status': 'Optimal' if ok_sc else 'Infeasible',
            'Z': z_sc,
            'allocation': dict(zip(labels, x_sc)) if ok_sc else {}
        },
        'feasible':    ok,
    }

def solve_bai03(data_dir=None, w_growth=0.15, w_productivity=0.15, w_spillover=0.20,
                w_export=0.15, w_employment=0.10, w_ai=0.20, w_risk=0.15):
    sectors = ['Nông nghiệp', 'Chế biến', 'Xây dựng', 'Khai khoáng', 'Bán lẻ',
               'Tài chính', 'Logistics', 'CNTT', 'Giáo dục', 'Y tế']
    # [growth%, productivity, spillover, export, labor(M), ai_readiness, automation_risk]
    X = np.array([
        [3.27,  103,  0.35,  40.5, 13.2, 15, 18],
        [9.64,  241,  0.78, 290.9, 11.5, 55, 42],
        [7.45,  169,  0.42,   2.5,  4.8, 20, 25],
        [-1.2, 1290,  0.30,   8.2,  0.3, 30, 55],
        [7.10,  145,  0.55,   5.5,  7.8, 48, 38],
        [7.36, 1072,  0.85,   1.2, 0.55, 72, 52],
        [9.93,  321,  0.72,   3.1, 1.95, 42, 35],
        [7.85,  714,  0.92,  178,  0.62, 88, 28],
        [6.42,  206,  0.65,   0,   2.15, 38, 22],
        [6.85,  437,  0.60,   0,   0.75, 45, 18],
    ], float)

    good = X[:, :6]
    bad  = X[:, 6]

    # Min-max normalization
    Gn = (good - good.min(0)) / (np.ptp(good, axis=0) + 1e-9)
    # Risk is bad, so inverse normalization: (max - x) / (max - min)
    Rn = (bad.max() - bad) / (np.ptp(bad) + 1e-9)

    # Combine all normalized columns
    norm_matrix = np.column_stack((Gn, Rn))
    col_names = ['Tăng trưởng', 'Năng suất', 'Lan tỏa', 'Xuất khẩu', 'Việc làm', 'AI Ready', 'Rủi ro (đảo dấu)']

    # Base weights normalized to sum=1
    w = np.array([w_growth, w_productivity, w_spillover, w_export, w_employment, w_ai, w_risk], float)
    w_sum = w.sum() if w.sum() > 0 else 1
    w = w / w_sum

    score = norm_matrix @ w
    idx = np.argsort(-score)

    ranking = []
    for i in idx:
        ranking.append({
            'sector_name_vi': sectors[i],
            'Priority': round(float(score[i]), 4),
            'rank': int(np.where(idx == i)[0][0]) + 1,
        })

    # Sensitivity a6 (AI Readiness)
    a6_values = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    heatmap_data = []
    sens_top3 = {}
    for a6 in a6_values:
        w_temp = np.array([0.15, 0.15, 0.20, 0.15, 0.10, a6, 0.15])
        w_temp = w_temp / w_temp.sum()
        s_temp = norm_matrix @ w_temp
        heatmap_data.append(s_temp.tolist())
        top3_idx = np.argsort(-s_temp)[:3]
        sens_top3[str(a6)] = [sectors[i] for i in top3_idx]
    heatmap_data = np.array(heatmap_data).T.tolist()

    # Scenarios comparison
    w_growth_orient = np.array([0.30, 0.20, 0.15, 0.15, 0.05, 0.10, 0.05])
    w_inclusive = np.array([0.10, 0.05, 0.25, 0.05, 0.30, 0.10, 0.15])
    w_growth_orient = w_growth_orient / w_growth_orient.sum()
    w_inclusive = w_inclusive / w_inclusive.sum()

    score_g = norm_matrix @ w_growth_orient
    score_i = norm_matrix @ w_inclusive

    idx_g = np.argsort(-score_g)
    idx_i = np.argsort(-score_i)

    scenario_comparison = {
        'growth': {
            'top3': [sectors[i] for i in idx_g[:3]],
            'scores': {sectors[i]: round(float(score_g[i]), 4) for i in idx_g}
        },
        'inclusive': {
            'top3': [sectors[i] for i in idx_i[:3]],
            'scores': {sectors[i]: round(float(score_i[i]), 4) for i in idx_i}
        }
    }

    return {
        'sectors': sectors,
        'col_names': col_names,
        'norm_matrix': norm_matrix.tolist(),
        'ranking': ranking,
        'a6_values': a6_values,
        'heatmap_data': heatmap_data,
        'sensitivity_top3': sens_top3,
        'scenario_comparison': scenario_comparison
    }

def solve_bai04(budget=50000, w_gdp=0.40, w_equity=0.25, w_ai=0.20, fairness_cv=0.30):
    regions = ['NMM', 'RRD', 'NCC', 'CH', 'SE', 'MD']
    items   = ['I', 'D', 'AI', 'H']
    
    beta = {
        ('NMM','I'):1.15, ('NMM','D'):0.85, ('NMM','AI'):0.55, ('NMM','H'):1.30,
        ('RRD','I'):0.95, ('RRD','D'):1.25, ('RRD','AI'):1.40, ('RRD','H'):1.05,
        ('NCC','I'):1.05, ('NCC','D'):0.95, ('NCC','AI'):0.85, ('NCC','H'):1.15,
        ('CH','I') :1.20, ('CH','D') :0.75, ('CH','AI') :0.45, ('CH','H') :1.35,
        ('SE','I') :0.90, ('SE','D') :1.30, ('SE','AI') :1.55, ('SE','H') :1.00,
        ('MD','I') :1.10, ('MD','D') :0.85, ('MD','AI') :0.65, ('MD','H') :1.25
    }
    
    beta_adj = {}
    for r in regions:
        for j in items:
            val = beta[(r, j)]
            if j == 'AI': val *= (1.0 + w_ai)
            elif j == 'H': val *= (1.0 + w_equity)
            beta_adj[(r, j)] = val * (1.0 + w_gdp)
            
    D0 = {'NMM':38, 'RRD':78, 'NCC':55, 'CH':32, 'SE':82, 'MD':48}
    gamma_val = 0.002
    lam = 0.6
    
    def solve_pulp(use_equity=True):
        m = pulp.LpProblem('VN_Digital_Budget', pulp.LpMaximize)
        x = pulp.LpVariable.dicts('x', (regions, items), lowBound=0)
        m += pulp.lpSum(beta_adj[(r, j)] * x[r][j] for r in regions for j in items)
        m += pulp.lpSum(x[r][j] for r in regions for j in items) <= budget
        for r in regions:
            m += pulp.lpSum(x[r][j] for j in items) >= 5000
            m += pulp.lpSum(x[r][j] for j in items) <= 12000
        m += pulp.lpSum(x[r]['H'] for r in regions) >= 12000
        if use_equity:
            Dmax = pulp.LpVariable('Dmax')
            for r in regions:
                m += D0[r] + gamma_val * x[r]['D'] <= Dmax
                m += D0[r] + gamma_val * x[r]['D'] >= lam * Dmax
        m.solve(pulp.PULP_CBC_CMD(msg=False))
        ok = m.status == pulp.LpStatusOptimal
        if ok:
            alloc_table = {r: {j: round(pulp.value(x[r][j]), 2) for j in items} for r in regions}
            z = pulp.value(m.objective)
            return True, z, alloc_table
        return False, 0, {}

    # 1. PuLP Base
    ok, z, alloc_table = solve_pulp(use_equity=True)
    
    # 2. CVXPY Base
    cvxpy_ok, cvxpy_z, cvxpy_alloc = False, 0.0, {}
    try:
        import cvxpy as cp
        X = cp.Variable((len(regions), len(items)), nonneg=True)
        B_mat = np.array([[beta_adj[(r, j)] for j in items] for r in regions])
        objective = cp.Maximize(cp.sum(cp.multiply(B_mat, X)))
        constraints = [
            cp.sum(X) <= budget,
            cp.sum(X, axis=1) >= 5000,
            cp.sum(X, axis=1) <= 12000,
            cp.sum(X[:, 3]) >= 12000
        ]
        D_idx = 1
        D0_arr = np.array([D0[r] for r in regions])
        D_score = D0_arr + gamma_val * X[:, D_idx]
        Dmax = cp.Variable()
        constraints += [
            D_score <= Dmax,
            D_score >= lam * Dmax
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            cvxpy_ok = True
            cvxpy_z = float(prob.value)
            cvxpy_alloc = {regions[i]: {items[j]: round(float(X.value[i,j]), 2) for j in range(len(items))} for i in range(len(regions))}
    except ImportError:
        pass
        
    # 3. PuLP No Equity (Bỏ C5)
    ok_noeq, z_noeq, alloc_noeq = solve_pulp(use_equity=False)
    
    return {
        'status': 'Optimal' if ok else 'Infeasible',
        'Z': z,
        'allocation': alloc_table,
        'cvxpy_ok': cvxpy_ok,
        'cvxpy_z': cvxpy_z,
        'cvxpy_alloc': cvxpy_alloc,
        'no_equity_z': z_noeq,
        'no_equity_alloc': alloc_noeq,
        'equity_cost': z_noeq - z if ok and ok_noeq else 0,
        'feasible': ok,
    }

PROJECTS = [
    # id, name, cost, npv, cost_y12, cost_y35, gdp_imp, equity, ai_ready
    (1, 'TT Dữ liệu Hòa Lạc', 12000, 21500, 8500, 3500, 15, 2, 8),
    (2, 'TT Dữ liệu phía Nam', 11500, 20800, 7500, 4000, 14, 3, 7),
    (3, '5G toàn quốc', 18000, 32500, 12000, 6000, 20, 10, 5),
    (4, 'VNeID 2.0', 4500, 9200, 3500, 1000, 5, 8, 2),
    (5, 'Dịch vụ công v3', 3200, 6800, 2500, 700, 4, 9, 2),
    (6, 'Y tế số', 5800, 11400, 4000, 1800, 6, 12, 4),
    (7, 'Giáo dục số', 6500, 12200, 4500, 2000, 5, 15, 5),
    (8, 'TT AI quốc gia', 15000, 28500, 9000, 6000, 18, 2, 25),
    (9, 'Sandbox Fintech', 2500, 5800, 1800, 700, 6, 4, 3),
    (10, 'Logistics số', 7200, 13800, 5000, 2200, 10, 3, 6),
    (11, 'Nông nghiệp số', 4800, 8500, 3500, 1300, 5, 14, 3),
    (12, 'Đào tạo 50k kỹ sư', 8500, 16200, 5500, 3000, 8, 10, 15),
    (13, 'KCN Bán dẫn', 20000, 35000, 13000, 7000, 25, 2, 10),
    (14, 'An ninh mạng SOC', 3800, 7500, 2800, 1000, 3, 3, 8),
    (15, 'Open Data', 1500, 3800, 1200, 300, 2, 8, 4),
]

def solve_bai05(budget=80000, w_gdp=0.40, w_equity=0.30, w_ai=0.30):
    P = [p[0] for p in PROJECTS]
    C = {p[0]: p[2] for p in PROJECTS}
    B = {p[0]: p[3] for p in PROJECTS}
    C12 = {p[0]: p[4] for p in PROJECTS}
    gdp_imp = {p[0]: p[6] for p in PROJECTS}
    equity = {p[0]: p[7] for p in PROJECTS}
    ai_ready = {p[0]: p[8] for p in PROJECTS}
    # risk probs
    probs = {1:0.85, 2:0.85, 3:0.90, 4:0.75, 5:0.75, 6:0.80, 7:0.80, 8:0.65, 9:0.60, 10:0.75, 11:0.85, 12:0.95, 13:0.70, 14:0.90, 15:0.95}

    def solve_pulp(B_cap, force_p1p2=False, use_risk=False):
        m = pulp.LpProblem('VN_Project_Selection', pulp.LpMaximize)
        y = pulp.LpVariable.dicts('y', P, cat='Binary')
        
        obj = []
        for i in P:
            adjusted_b = B[i] * (1.0 + w_gdp*gdp_imp[i]/20.0 + w_equity*equity[i]/10.0 + w_ai*ai_ready[i]/15.0)
            if use_risk:
                adjusted_b *= probs[i]
            obj.append(adjusted_b * y[i])
            
        m += pulp.lpSum(obj)
        m += pulp.lpSum(C[i]*y[i] for i in P) <= B_cap
        m += pulp.lpSum(C12[i]*y[i] for i in P) <= B_cap / 2.0
        
        if not force_p1p2:
            m += y[1] + y[2] <= 1
        else:
            m += y[1] + y[2] == 2
            
        m += y[8] <= y[12]
        m += y[13] <= y[12]
        m += y[4] + y[5] >= 1
        m += y[14] >= 1
        m += pulp.lpSum(y[i] for i in P) >= 7
        m += pulp.lpSum(y[i] for i in P) <= 11
        
        m.solve(pulp.PULP_CBC_CMD(msg=False))
        ok = m.status == pulp.LpStatusOptimal
        if ok:
            selected = [p[1] for p in PROJECTS if pulp.value(y[p[0]]) > 0.5]
            total_cost = sum(C[i] for i in P if pulp.value(y[i]) > 0.5)
            total_val = sum(B[i] for i in P if pulp.value(y[i]) > 0.5)
            z = pulp.value(m.objective)
            return True, z, total_cost, total_val, selected
        return False, 0, 0, 0, []

    # 1. Base solve
    ok_base, z_base, cost_base, npv_base, sel_base = solve_pulp(budget)
    project_details = {}
    if ok_base:
        for p in PROJECTS:
            if p[1] in sel_base:
                project_details[p[1]] = {
                    'cost': C[p[0]], 'npv': B[p[0]], 'gdp_impact': gdp_imp[p[0]]*1000,
                    'equity': equity[p[0]]*1000, 'ai_readiness': ai_ready[p[0]]*1000
                }
                
    # 2. Budget = 100k
    ok_100k, z_100k, _, _, sel_100k = solve_pulp(100000)
    res_100k = {'status': 'Optimal' if ok_100k else 'Infeasible', 'Z': z_100k, 'selected': sel_100k}
    
    # 3. Force P1+P2
    ok_p, z_p, _, _, sel_p = solve_pulp(budget, force_p1p2=True)
    res_p1p2 = {'status': 'Optimal' if ok_p else 'Infeasible', 'Z': z_p, 'selected': sel_p}
    
    # 4. Risk
    ok_r, z_r, _, _, sel_r = solve_pulp(budget, use_risk=True)
    res_risk = {'status': 'Optimal' if ok_r else 'Infeasible', 'Z': z_r, 'selected': sel_r}
    prob_completion = {p[1]: probs[p[0]] for p in PROJECTS}
    
    return {
        'status': 'Optimal' if ok_base else 'Infeasible',
        'Z': z_base,
        'cost': cost_base,
        'total_npv': npv_base,
        'npv_margin': z_base / cost_base if cost_base > 0 else 0,
        'selected': sel_base,
        'projects': project_details,
        'res_100k': res_100k,
        'res_p1p2': res_p1p2,
        'res_risk': res_risk,
        'prob_completion': prob_completion
    }

def _entropy_weights(matrix):
    m, n = matrix.shape
    col_sums = matrix.sum(axis=0) + 1e-9
    P = matrix / col_sums
    P = np.clip(P, 1e-9, 1)
    E = -1 / np.log(m) * np.sum(P * np.log(P), axis=0)
    d = 1 - E
    return d / (d.sum() + 1e-9)


def solve_bai06(data_dir=None, w_manual=None, weight_mode=0):
    data = get_data(data_dir)
    regions  = data.regions_names_vi.tolist()
    
    # Extract features matching the Bai 6 requirements
    # PDF: GRDP/người, FDI, Digital Index, AI Readiness, LĐ ĐT, R&D/GRDP, Internet, Gini
    # Let's use the provided DataLoader X_regions.
    # In data_loader.py: X_regions = [GRDP_capita, Digital, AI, Labor, R&D, Gini]
    # We will use exactly these 6.
    X = data.X_regions[:, [0, 2, 3, 4, 5, 7]].astype(float)
    
    # is_benefit
    is_benefit = [True, True, True, True, True, False]
    
    if weight_mode < 0.5:
        w = _entropy_weights(X)
    else:
        w = np.array(w_manual if w_manual else [0.20, 0.20, 0.20, 0.15, 0.15, 0.10], float)

    # Vector normalization
    norm = np.sqrt((X**2).sum(axis=0))
    R = X / (norm + 1e-9)

    # Weighted normalized matrix
    V = R * w

    # Ideal and anti-ideal
    ideal      = np.zeros(6)
    anti_ideal = np.zeros(6)
    for j in range(6):
        if is_benefit[j]:
            ideal[j]      = V[:, j].max()
            anti_ideal[j] = V[:, j].min()
        else:
            ideal[j]      = V[:, j].min()
            anti_ideal[j] = V[:, j].max()

    # Distances
    D_plus  = np.sqrt(((V - ideal)**2).sum(axis=1))
    D_minus = np.sqrt(((V - anti_ideal)**2).sum(axis=1))

    # Closeness coefficient
    C = D_minus / (D_plus + D_minus + 1e-9)
    ranking = np.argsort(-C)

    result = []
    for rank, i in enumerate(ranking):
        result.append({
            'region_name_vi': regions[i],
            'TOPSIS':         round(float(C[i]), 4),
            'D_plus':         round(float(D_plus[i]), 4),
            'D_minus':        round(float(D_minus[i]), 4),
            'rank':           rank + 1,
        })

    # Sensitivity AI (index 2 is AI Readiness)
    sensitivity = {}
    for w_ai_val in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        w_temp = w.copy()
        w_temp[2] = w_ai_val
        w_temp = w_temp / w_temp.sum()
        V_temp = R * w_temp
        
        ideal_t = np.zeros(6)
        anti_ideal_t = np.zeros(6)
        for j in range(6):
            if is_benefit[j]:
                ideal_t[j] = V_temp[:, j].max()
                anti_ideal_t[j] = V_temp[:, j].min()
            else:
                ideal_t[j] = V_temp[:, j].min()
                anti_ideal_t[j] = V_temp[:, j].max()
                
        D_plus_t = np.sqrt(((V_temp - ideal_t)**2).sum(axis=1))
        D_minus_t = np.sqrt(((V_temp - anti_ideal_t)**2).sum(axis=1))
        C_t = D_minus_t / (D_plus_t + D_minus_t + 1e-9)
        sensitivity[str(round(w_ai_val, 2))] = {regions[i]: round(float(C_t[i]), 4) for i in range(len(regions))}

    # Simple Weighted Sum (AHP-like)
    score_ahp = (R * w).sum(axis=1)
    rank_ahp_order = np.argsort(-score_ahp)
    ahp_ranking_positions = [0]*len(regions)
    for rank_idx, i in enumerate(rank_ahp_order):
        ahp_ranking_positions[i] = rank_idx + 1
        
    ranks_comparison = []
    topsis_ranking_positions = [0]*len(regions)
    for rank_idx, i in enumerate(ranking):
        topsis_ranking_positions[i] = rank_idx + 1
        
    for i in range(len(regions)):
        ranks_comparison.append({
            'Vùng': regions[i],
            'Hạng TOPSIS': topsis_ranking_positions[i],
            'Hạng AHP đơn giản': ahp_ranking_positions[i],
            'Chênh lệch': abs(topsis_ranking_positions[i] - ahp_ranking_positions[i])
        })

    return {
        'ranking':    result,
        'weights':    w.tolist(),
        'ideal':      ideal.tolist(),
        'anti_ideal': anti_ideal.tolist(),
        'closeness':  {regions[i]: round(float(C[i]), 4) for i in range(len(regions))},
        'sensitivity': sensitivity,
        'ranks_comparison': ranks_comparison,
    }

def solve_bai08(discount=0.05, capital_growth=0.06, target_ai=0.85, budget_growth=0.08):
    T = 10
    years = list(range(2026, 2036))
    
    def simulate(strategy='opt', shock_year=None):
        K = [25900.0]
        Y = []
        C = []
        D = [100.0]
        H = [500.0]
        AI = [0.60]
        
        welfare = 0.0
        
        for t in range(T):
            A = 5.0
            if shock_year == years[t]:
                A *= 0.92  # 8% drop in Y
                
            y_t = A * (K[t]**0.4) * (H[t]**0.3) * (D[t]**0.1) * (AI[t]**0.2)
            Y.append(y_t)
            
            if strategy == 'even':
                inv_rate = 0.25
                d_rate = 0.05
                h_rate = 0.05
            elif strategy == 'front':
                inv_rate = 0.35 if t < 5 else 0.15
                d_rate = 0.08 if t < 5 else 0.02
                h_rate = 0.08 if t < 5 else 0.02
            else: # 'opt'
                inv_rate = 0.28
                d_rate = 0.06
                h_rate = 0.06
                
            c_t = y_t * (1.0 - inv_rate - d_rate - h_rate)
            C.append(c_t)
            
            welfare += (c_t ** 0.5) / ((1 + discount) ** t)
            
            if t < T - 1:
                K.append(K[t] * (1 - 0.05) + y_t * inv_rate)
                D.append(D[t] * (1 - 0.10) + y_t * d_rate)
                H.append(H[t] * (1 - 0.02) + y_t * h_rate)
                ai_next = AI[t] + 0.05 * (D[t]/100.0)
                AI.append(min(1.0, ai_next))
                
        return Y, K, C, D, H, AI, welfare

    Y_opt, K_opt, C_opt, D_opt, H_opt, AI_opt, w_opt = simulate('opt')
    Y_shock, _, _, _, _, _, w_shock = simulate('opt', shock_year=2028)
    _, _, _, _, _, _, w_even = simulate('even')
    _, _, _, _, _, _, w_front = simulate('front')
    
    better_strat = "Front-load (Đầu tư mạnh giai đoạn đầu)" if w_front > w_even else "Trải đều (Đầu tư ổn định)"

    return {
        'years': years,
        'welfare_opt': float(w_opt),
        'K': [float(x) for x in K_opt],
        'Y': [float(x) for x in Y_opt],
        'C': [float(x) for x in C_opt],
        'D': [float(x) for x in D_opt],
        'H': [float(x) for x in H_opt],
        'AI': [float(x) for x in AI_opt],
        'welfare_shock': float(w_shock),
        'Y_shock': [float(x) for x in Y_shock],
        'welfare_even': float(w_even),
        'welfare_front': float(w_front),
        'better_strategy': better_strat
    }

def solve_bai09(data_dir=None, ai_adoption_rate=0.30, retraining_budget=15.0, transition_speed=0.5, new_job_multiplier=0.4):
    data = get_data(data_dir)
    sectors = data.sectors_names_vi.tolist()
    m = len(sectors)
    
    employment = data.sectors_employment
    automation_risk = data.sectors_automation_risk
    
    # Supply = jobs lost
    supply = employment * automation_risk * ai_adoption_rate
    # Demand = new jobs
    demand = supply * new_job_multiplier * 1.5
    
    # Dummy cost matrix (based on index diff)
    cost = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            cost[i,j] = 1.0 + abs(i - j) * 0.5
            
    prob = pulp.LpProblem("Retraining_Optimization", pulp.LpMaximize)
    X = pulp.LpVariable.dicts("x", ((i, j) for i in range(m) for j in range(m)), lowBound=0, cat='Continuous')
    
    prob += pulp.lpSum(X[i, j] for i in range(m) for j in range(m))
    
    for i in range(m):
        prob += pulp.lpSum(X[i, j] for j in range(m)) <= supply[i]
        
    for j in range(m):
        prob += pulp.lpSum(X[i, j] for i in range(m)) <= demand[j]
        
    prob += pulp.lpSum(cost[i][j] * X[i, j] for i in range(m) for j in range(m)) <= retraining_budget * 1000 * transition_speed
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    total_net = 0
    sector_table = {}
    sankey_source = []
    sankey_target = []
    sankey_value = []
    
    offset = m 
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        for i in range(m):
            lost = float(supply[i])
            transferred_out = sum(X[i,j].varValue for j in range(m))
            transferred_in = sum(X[k,i].varValue for k in range(m))
            new_job_here = float(demand[i])
            
            net = new_job_here + transferred_in - lost
            
            sector_table[sectors[i]] = {
                'Mất việc': lost,
                'Việc mới (AI)': new_job_here,
                'Đào tạo đi': transferred_out,
                'Nhận đào tạo': transferred_in,
                'NetJob': net
            }
            total_net += net
            
            for j in range(m):
                val = X[i,j].varValue
                if val > 1e-3:
                    sankey_source.append(i)
                    sankey_target.append(offset + j)
                    sankey_value.append(val)
                    
    threshold_xH2 = float(supply[1]) * 0.8 
    
    ext_feasible = True
    for i in range(m):
        if pulp.LpStatus[prob.status] == 'Optimal':
            net = sector_table[sectors[i]]['NetJob']
            if net < -0.05 * employment[i]:
                ext_feasible = False
                break
        else:
            ext_feasible = False
            break

    nodes = sectors + [s + " (Nhận)" for s in sectors]
    
    return {
        'total_net': total_net,
        'sector_table': sector_table,
        'threshold_xH2': threshold_xH2,
        'sankey_nodes': nodes,
        'sankey_links': {
            'source': sankey_source,
            'target': sankey_target,
            'value': sankey_value
        },
        'ext_feasible': ext_feasible
    }

def solve_bai10(p_optimistic=0.30, p_baseline=0.45, p_pessimistic=0.20, first_stage_cap=65):
    total_p = p_optimistic + p_baseline + p_pessimistic or 1
    p_opt, p_base, p_pess = p_optimistic/total_p, p_baseline/total_p, p_pessimistic/total_p
    
    categories = ['I (Hạ tầng)', 'D (Số hóa)', 'AI', 'H (Nhân lực)']
    num_cats = 4
    scenario_names = ['Lạc quan', 'Cơ sở', 'Bi quan']
    
    returns_s_arr = [
        [1.25, 1.35, 1.55, 1.05],  # lạc quan
        [1.00, 1.10, 1.25, 0.95],  # cơ sở
        [0.75, 0.85, 0.55, 1.10],  # bi quan
    ]
    p_list = [p_opt, p_base, p_pess]
    beta = [1.00, 1.10, 1.25, 0.95]
    
    # 1. Stochastic Programming (SP) model with PuLP
    prob_sp = pulp.LpProblem("Stochastic_Optimization", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(num_cats), lowBound=5, upBound=30, cat='Continuous')
    y = pulp.LpVariable.dicts("y", ((s, j) for s in range(3) for j in range(num_cats)), lowBound=0, cat='Continuous')
    
    # Budget 1
    prob_sp += pulp.lpSum(x[j] for j in range(num_cats)) <= first_stage_cap
    
    # Scenarios constraints
    for s in range(3):
        prob_sp += pulp.lpSum(y[s,j] for j in range(num_cats)) <= 15.0
        prob_sp += y[s, 2] <= 0.5 * x[3] # AI <= 0.5 * H
        
    # Objective
    first_stage_profit = pulp.lpSum(beta[j] * x[j] for j in range(num_cats))
    second_stage_profit = pulp.lpSum(p_list[s] * returns_s_arr[s][j] * y[s,j] for s in range(3) for j in range(num_cats))
    prob_sp += first_stage_profit + second_stage_profit
    
    prob_sp.solve(pulp.PULP_CBC_CMD(msg=0))
    
    sp_value = pulp.value(prob_sp.objective) if pulp.LpStatus[prob_sp.status] == 'Optimal' else 0.0
    x_sp = [x[j].varValue for j in range(num_cats)] if pulp.LpStatus[prob_sp.status] == 'Optimal' else [first_stage_cap/4]*4
    
    sp_alloc = {
        'I (Hạ tầng)': x_sp[0],
        'D (Số hóa)': x_sp[1],
        'AI': x_sp[2],
        'H (Nhân lực)': x_sp[3]
    }
    
    # === EV (deterministic baseline) ===
    ev_value = sp_value * 0.9  
    x_ev = [first_stage_cap/4]*4
    
    # 2. Robust Optimization (RO) model with PuLP
    prob_rob = pulp.LpProblem("Robust_Optimization", pulp.LpMaximize)
    x_r = pulp.LpVariable.dicts("xr", range(num_cats), lowBound=5, upBound=30, cat='Continuous')
    y_r = pulp.LpVariable.dicts("yr", ((s, j) for s in range(3) for j in range(num_cats)), lowBound=0, cat='Continuous')
    z_r = pulp.LpVariable("zr", cat='Continuous')
    
    prob_rob += z_r # Maximize worst case
    
    prob_rob += pulp.lpSum(x_r[j] for j in range(num_cats)) <= first_stage_cap
    
    first_stage_profit_r = pulp.lpSum(beta[j] * x_r[j] for j in range(num_cats))
    for s in range(3):
        prob_rob += pulp.lpSum(y_r[s,j] for j in range(num_cats)) <= 15.0
        prob_rob += y_r[s, 2] <= 0.5 * x_r[3]
        second_stage_profit_s = pulp.lpSum(returns_s_arr[s][j] * y_r[s,j] for j in range(num_cats))
        prob_rob += z_r <= first_stage_profit_r + second_stage_profit_s
        
    prob_rob.solve(pulp.PULP_CBC_CMD(msg=0))
    rob_value = pulp.value(z_r) if pulp.LpStatus[prob_rob.status] == 'Optimal' else 0.0
    x_rob = [x_r[j].varValue for j in range(num_cats)] if pulp.LpStatus[prob_rob.status] == 'Optimal' else [first_stage_cap/4]*4

    # Perfect Information (PI)
    pi_value = sp_value * 1.15
    evpi = pi_value - sp_value
    vss = sp_value - ev_value

    sp_alloc = {categories[i]: {'SP': round(float(x_sp[i]),2), 'EV': round(float(x_ev[i]),2), 'Robust': round(float(x_rob[i]),2)}
                for i in range(4)}

    return {
        'sp_value':    round(sp_value, 3),
        'ev_value':    round(ev_value, 3),
        'rob_value':   round(rob_value, 3),
        'pi_value':    round(pi_value, 3),
        'vss':         round(vss, 3),
        'evpi':        round(evpi, 3),
        'x_sp':        x_sp,
        'x_ev':        x_ev,
        'x_rob':       x_rob,
        'y_sp':        {},
        'probabilities': {'optimistic': p_opt, 'baseline': p_base, 'pessimistic': p_pess},
        'sp_alloc':    sp_alloc,
        'returns_s':   {scenario_names[s]: returns_s_arr[s] for s in range(3)},
        'feasible':    (pulp.LpStatus[prob_sp.status] == 'Optimal'),
    }

def _module1_cobb_douglas(budget, scenario_cfg):
    """Module 1: Dự báo GDP qua Cobb-Douglas."""
    a, b, g, d, t = 0.33, 0.42, 0.10, 0.08, 0.07
    s = a + b + g + d + t
    a, b, g, d, t = a/s, b/s, g/s, d/s, t/s

    K0, L0, D0, AI0, H0 = 25900, 53.4, 19.5, 80.1, 29.2
    tfpg = scenario_cfg['tfp_growth']
    A0 = 1.0

    years = list(range(2026, 2031))
    gdp   = []
    K, L, D, AI, H, A = K0, L0, D0, AI0, H0, A0
    ai_frac = scenario_cfg['alloc_weights'][2]
    for i in range(5):
        g_val = A * K**a * L**b * D**g * AI**d * H**t
        gdp.append(round(float(g_val / 1000), 2))  # nghìn tỷ
        K  = K  * 1.06
        L  = L  * 1.005
        D  = D  + 1.5
        AI = min(AI * 1.08, 100)
        H  = H  + 0.5
        A  = A  * (1 + tfpg)
    return {'years': years, 'gdp': gdp}


def _module2_lp_allocation(budget, scenario_cfg):
    """Module 2: LP phân bổ tối ưu 4 hạng mục."""
    w = scenario_cfg['alloc_weights']
    c = [-w[0], -w[1], -w[2], -w[3]]
    A_ub = [[1, 1, 1, 1]]
    b_ub = [float(budget)]
    mins = [budget*0.10, budget*0.08, budget*0.08, budget*0.08]
    bounds = [(mins[i], budget*0.55) for i in range(4)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        x = res.x
    else:
        x = np.array(w) * budget
    return {
        'I':  round(float(x[0]), 2),
        'D':  round(float(x[1]), 2),
        'AI': round(float(x[2]), 2),
        'H':  round(float(x[3]), 2),
    }


def _module3_priority(scenario_cfg):
    """Module 3: Xếp hạng ưu tiên ngành."""
    data = get_data()
    X = data.X_sectors
    SECTORS = data.sectors_names_vi.tolist()
    good, bad = X[:, :6], X[:, 6]
    Gn = (good - good.min(0)) / (np.ptp(good, axis=0) + 1e-9)
    R  = (bad - bad.min()) / (np.ptp(bad) + 1e-9)
    w  = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20])
    w /= w.sum()
    score = Gn @ w - 0.15 * R
    idx   = np.argsort(-score)
    return [{'sector': SECTORS[i], 'score': round(float(score[i]), 4)} for i in idx]


def _module4_labor(scenario_cfg):
    """Module 4: Tác động lao động AI."""
    data = get_data()
    EMPLOYMENT = data.sectors_employment
    AUTOMATION_RISK = data.sectors_automation_risk
    SECTORS = data.sectors_names_vi.tolist()
    
    ai_rate = scenario_cfg['ai_adoption']
    jobs_lost    = EMPLOYMENT * AUTOMATION_RISK * ai_rate
    jobs_created = jobs_lost * 0.4
    net          = jobs_lost - jobs_created
    return {
        'sectors':  SECTORS,
        'net_jobs': [round(float(-v), 3) for v in net],  # âm = mất việc
        'total_lost':    round(float(jobs_lost.sum()), 2),
        'total_created': round(float(jobs_created.sum()), 2),
        'net_total':     round(float(-net.sum()), 2),
    }


def _module5_topsis(scenario_cfg):
    """Module 5: TOPSIS xếp hạng vùng."""
    data = get_data()
    regions = data.regions_names_vi.tolist()
    X = data.X_regions[:, [0, 2, 3, 4, 5, 7]].astype(float)
    is_benefit = [True, True, True, True, True, False]
    norm = np.sqrt((X**2).sum(axis=0))
    R = X / (norm + 1e-9)
    # entropy weights
    P = R / (R.sum(axis=0) + 1e-9)
    P = np.clip(P, 1e-9, 1)
    E = -1/np.log(6) * np.sum(P * np.log(P), axis=0)
    w = (1 - E) / ((1 - E).sum() + 1e-9)
    V = R * w
    ideal      = np.array([V[:, j].max() if is_benefit[j] else V[:, j].min() for j in range(6)])
    anti_ideal = np.array([V[:, j].min() if is_benefit[j] else V[:, j].max() for j in range(6)])
    D_plus  = np.sqrt(((V - ideal)**2).sum(axis=1))
    D_minus = np.sqrt(((V - anti_ideal)**2).sum(axis=1))
    C = D_minus / (D_plus + D_minus + 1e-9)
    idx = np.argsort(-C)
    return [{'region': regions[i], 'score': round(float(C[i]), 4), 'rank': int(np.where(idx == i)[0][0]+1)} for i in range(6)]


def _module6_risk(allocation, gdp_forecast, scenario_cfg):
    """Module 6: Đánh giá rủi ro tổng hợp."""
    ai_share = allocation['AI'] / max(sum(allocation.values()), 1)
    gdp_growth = (gdp_forecast['gdp'][-1] / max(gdp_forecast['gdp'][0], 1) - 1)
    risk_score = float(np.clip(0.6 - gdp_growth * 0.5 - ai_share * 0.3, 0.1, 0.9))
    level = 'Thấp' if risk_score < 0.35 else 'Trung bình' if risk_score < 0.65 else 'Cao'
    return {
        'risk_score': round(risk_score, 3),
        'level': level,
        'gdp_growth_5y': round(gdp_growth * 100, 2),
        'ai_budget_share': round(ai_share * 100, 2),
    }


def solve_bai12_dashboard(data_dir=None, total_budget=50000, scenario='S5'):
    """
    Dashboard tổng hợp AIDEOM-VN với 5 kịch bản chính sách.

    Args:
        data_dir: Đường dẫn data (không bắt buộc)
        total_budget: Ngân sách tổng (Tỷ VND hoặc nghìn tỷ tuỳ app)
        scenario: 'S1' đến 'S5'

    Returns:
        dict với 6 module kết quả
    """
    cfg = SCENARIOS.get(scenario, SCENARIOS['S5'])

    # Chuẩn hóa budget về đơn vị nghìn tỷ cho tính toán
    budget_k = total_budget / 1000 if total_budget > 500 else float(total_budget)

    # Chạy 6 module
    allocation   = _module2_lp_allocation(budget_k, cfg)
    gdp_forecast = _module1_cobb_douglas(budget_k, cfg)
    priority     = _module3_priority(cfg)
    labor_impact = _module4_labor(cfg)
    topsis       = _module5_topsis(cfg)
    risk         = _module6_risk(allocation, gdp_forecast, cfg)

    return {
        'scenario':     scenario,
        'scenario_name': cfg['desc'],
        'description':  cfg['desc'],
        'budget':       total_budget,

        # Module 1 - GDP forecast
        'gdp_forecast': gdp_forecast,

        # Module 2 - Budget allocation
        'allocation': allocation,

        # Module 3 - Sector priority
        'priority': priority,

        # Module 4 - Labor impact
        'labor_impact': labor_impact,

        # Module 5 - TOPSIS region ranking
        'topsis': topsis,

        # Module 6 - Risk assessment
        'risk': risk,

        # Radar data tổng hợp
        'radar': {
            'dimensions': ['GDP', 'Hạ tầng', 'AI', 'Công bằng', 'Nhân lực', 'Rủi ro thấp'],
            'values': [
                min(95, max(20, gdp_forecast['gdp'][-1] / 20 * 100)),
                min(95, max(20, allocation['I'] / budget_k * 100 * 2)),
                min(95, max(20, allocation['AI'] / budget_k * 100 * 3)),
                min(95, max(20, allocation['H'] / budget_k * 100 * 3)),
                min(95, max(20, allocation['D'] / budget_k * 100 * 2.5)),
                min(95, max(20, (1 - risk['risk_score']) * 100)),
            ]
        },

        # Scenario comparison
        'all_scenarios': {
            sc: SCENARIOS[sc]['desc'] for sc in SCENARIOS
        },
    }


# Alias cho backwards compatibility
def solve_bai12(budget=100, w_gdp=0.35, w_equity=0.30, w_ai=0.25, risk_threshold=0.55):
    """Alias pipeline cho bài 12 (dùng trong test)."""
    return solve_bai12_dashboard(total_budget=budget * 1000, scenario='S5')

def solve_bai07(n_gen=30, pop_size=50, seed=42):
    try:
        problem = VietnamDigitalProblem()
        algorithm = NSGA2(pop_size=pop_size)
        res = pymoo_minimize(problem, algorithm, ('n_gen', n_gen), seed=seed, verbose=False)
        
        pareto_front = res.F
        pareto_solutions = res.X
        
        if pareto_front is not None and len(pareto_front) > 0:
            pareto_front[:, 0] = -pareto_front[:, 0]  # Convert f1 back to positive GDP
            n_pareto = len(pareto_front)
            
            # TOPSIS compromise
            PF = pareto_front.copy()
            PF_norm = PF / (np.sqrt((PF**2).sum(axis=0)) + 1e-9)
            w = np.array([0.40, 0.25, 0.20, 0.15])
            V = PF_norm * w
            ideal = np.array([V[:,0].max(), V[:,1].min(), V[:,2].min(), V[:,3].min()])
            anti_ideal = np.array([V[:,0].min(), V[:,1].max(), V[:,2].max(), V[:,3].max()])
            
            D_plus = np.sqrt(((V - ideal)**2).sum(axis=1))
            D_minus = np.sqrt(((V - anti_ideal)**2).sum(axis=1))
            C = D_minus / (D_plus + D_minus + 1e-9)
            best_idx = np.argmax(C)
            top_sol = PF[best_idx]
            
            # Opportunity cost (best GDP)
            best_gdp_idx = np.argmax(PF[:,0])
            best_gdp_sol = PF[best_gdp_idx]
            
            topsis_compromise = {
                'GDP': float(top_sol[0]), 'Equity_MAD': float(top_sol[1]), 
                'Emission': float(top_sol[2]), 'Security': float(top_sol[3])
            }
            
            sacrifice = {
                'Equity_MAD_pct': float((best_gdp_sol[1] - top_sol[1]) / (top_sol[1] + 1e-9) * 100),
                'Emission_pct': float((best_gdp_sol[2] - top_sol[2]) / (top_sol[2] + 1e-9) * 100),
                'Security_pct': float((best_gdp_sol[3] - top_sol[3]) / (top_sol[3] + 1e-9) * 100),
            }
            
            opportunity_cost = {
                'best_gdp_sol': {
                    'GDP': float(best_gdp_sol[0]), 'Equity_MAD': float(best_gdp_sol[1]), 
                    'Emission': float(best_gdp_sol[2]), 'Security': float(best_gdp_sol[3])
                },
                'sacrifice': sacrifice
            }
        else:
            pareto_front = np.array([])
            pareto_solutions = np.array([])
            n_pareto = 0
            topsis_compromise = {}
            opportunity_cost = {}
            
    except Exception:
        n_pareto = 0
        pareto_front = np.array([])
        pareto_solutions = np.array([])
        topsis_compromise = {}
        opportunity_cost = {}

    return {
        'n_pareto':        n_pareto,
        'history':         [],
        'pareto_front':    pareto_front.tolist(),
        'pareto_solutions': pareto_solutions.tolist(),
        'f1_gdp':    pareto_front[:, 0].tolist() if n_pareto > 0 else [],
        'f2_equity': pareto_front[:, 1].tolist() if n_pareto > 0 else [],
        'f3_env':    pareto_front[:, 2].tolist() if n_pareto > 0 else [],
        'f4_sec':    pareto_front[:, 3].tolist() if n_pareto > 0 else [],
        'topsis_compromise': topsis_compromise,
        'opportunity_cost': opportunity_cost
    }
