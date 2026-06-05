from django.shortcuts import render
import sys
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html

# Add the root directory to path to import src
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.optimization import solve_bai01, solve_bai02, solve_bai03, solve_bai04, solve_bai05, solve_bai06, solve_bai07, solve_bai08, solve_bai09, solve_bai10, solve_bai12, solve_bai12_dashboard
from src.rl_env import solve_bai11

DATA_DIR = os.path.join(root_path, 'data')

BAI_LIST = [
    {'id': 1,  'label': 'Bài 1: Cobb-Douglas'},
    {'id': 2,  'label': 'Bài 2: LP Ngân sách'},
    {'id': 3,  'label': 'Bài 3: Chỉ số ưu tiên'},
    {'id': 4,  'label': 'Bài 4: LP Vùng'},
    {'id': 5,  'label': 'Bài 5: MIP Dự án'},
    {'id': 6,  'label': 'Bài 6: TOPSIS'},
    {'id': 7,  'label': 'Bài 7: Đa mục tiêu'},
    {'id': 8,  'label': 'Bài 8: Tối ưu động'},
    {'id': 9,  'label': 'Bài 9: Mô phỏng lao động'},
    {'id': 10, 'label': 'Bài 10: QH ngẫu nhiên'},
    {'id': 11, 'label': 'Bài 11: Q-learning'},
    {'id': 12, 'label': 'Bài 12: Đồ án tổng hợp'},
]

PLOT_LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#111827', family='Inter, sans-serif'))
LINE_COLOR = '#2563EB'

def _fig_to_html(fig):
    fig.update_layout(**PLOT_LAYOUT)
    return to_html(fig, full_html=False, include_plotlyjs='cdn')

def home(request):
    current_bai = int(request.GET.get('bai', 12))
    scenario = request.GET.get('scenario', 'S5')
    budget = float(request.GET.get('budget', 50000))

    ctx = {
        'bai_list': BAI_LIST,
        'current_bai': current_bai,
        'scenario': scenario,
        'budget': budget,
        'title': next((b['label'] for b in BAI_LIST if b['id'] == current_bai), 'AIDEOM-VN'),
        'graph_html': None,
        'extra_info': None,
    }

    if current_bai == 1:
        alpha = float(request.GET.get('alpha', 0.33))
        beta = float(request.GET.get('beta', 0.42))
        gamma_val = float(request.GET.get('gamma', 0.10))
        delta = float(request.GET.get('delta', 0.08))
        theta = float(request.GET.get('theta', 0.07))
        res = solve_bai01(DATA_DIR, alpha, beta, gamma_val, delta, theta)
        
        html_parts = []
        
        # 1. TFP
        fig_at = px.line(x=res['years'], y=res['A_t'], title="1. Xu hướng TFP (A_t)", markers=True)
        fig_at.update_traces(line_color=LINE_COLOR)
        html_parts.append(_fig_to_html(fig_at))
        
        # 2. Compare Y
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(x=res['years'], y=res['Y_actual'], name='Y thực tế', mode='lines+markers', line=dict(color='#E53E3E')))
        if 'Y_hat' in res:
            fig_compare.add_trace(go.Scatter(x=res['years'], y=res['Y_hat'], name='Ŷ dự báo', mode='lines+markers', line=dict(color=LINE_COLOR, dash='dash')))
        fig_compare.update_layout(title=f"2. So sánh Y thực tế vs Ŷ dự báo (MAPE: {res['mape']:.2f}%)")
        html_parts.append(_fig_to_html(fig_compare))
        
        # 3. Decomp
        contrib = res['contrib_pct']
        import pandas as pd
        df_contrib = pd.DataFrame({'Yếu tố': list(contrib.keys()), 'Đóng góp (%)': list(contrib.values())})
        fig_decomp = px.bar(df_contrib, x='Yếu tố', y='Đóng góp (%)', title="3. Phân rã tăng trưởng GDP 2020-2025 (%)", color='Yếu tố')
        html_parts.append(_fig_to_html(fig_decomp))
        
        # 4. Forecast
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=res['years'], y=res['Y_actual'], name='Y thực tế', mode='lines+markers', line=dict(color='#E53E3E')))
        fig_forecast.add_trace(go.Scatter(x=res['forecast_years'], y=res['forecast_series'], name='Dự báo cơ sở', mode='lines+markers', line=dict(color=LINE_COLOR)))
        fig_forecast.update_layout(title=f"4. Dự báo GDP đến 2030 (Cơ sở: {res['gdp_2030']:,.0f} nghìn tỷ VND)")
        html_parts.append(_fig_to_html(fig_forecast))
        
        ctx.update({
            'graph_html': "".join(f"<div style='margin-bottom:30px;'>{part}</div>" for part in html_parts),
            'mape': f"{res['mape']:.2f}",
            'alpha': alpha, 'beta': beta, 'gamma_val': gamma_val, 'delta': delta, 'theta': theta
        })

    elif current_bai == 2:
        tb = int(request.GET.get('total_budget', 100))
        mi = int(request.GET.get('min_I', 25))
        mai = int(request.GET.get('min_AI', 15))
        res = solve_bai02(budget=tb, min_I=mi, min_AI=mai)
        
        html_parts = []
        if res['status'] == 'Optimal':
            # 1. Linprog
            labels = list(res['allocation'].keys())
            values = list(res['allocation'].values())
            fig1 = px.bar(x=labels, y=values, title=f"1. Phân bổ tối ưu (linprog) - Z*={res['Z']:.2f}", color=labels)
            html_parts.append(_fig_to_html(fig1))
            
            # 2. Dual values
            dv = res.get('dual_values', {})
            dv_html = f"<h3>2. Giá đối ngẫu (Shadow Prices)</h3><table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%; text-align: left; margin-bottom: 10px;'><tr><th style='padding:8px; border: 1px solid #ddd;'>Ràng buộc</th><th style='padding:8px; border: 1px solid #ddd;'>Shadow Price (π)</th></tr>"
            for k, v in dv.items():
                dv_html += f"<tr><td style='padding:8px; border: 1px solid #ddd;'>{k}</td><td style='padding:8px; border: 1px solid #ddd;'>{v}</td></tr>"
            dv_html += "</table>"
            budget_sp = dv.get('C1_Budget', 0)
            dv_html += f"<p><b>Ý nghĩa chính sách:</b> Shadow price của C1_Budget = <b>{budget_sp}</b>. Nếu tăng thêm 1 đơn vị ngân sách, Z* tăng thêm {budget_sp} đơn vị.</p>"
            html_parts.append(dv_html)
            
            # 3. Sensitivity
            fig3 = px.line(x=res['sensitivity_budgets'], y=res['sensitivity_z'], title="3. Phân tích độ nhạy – Đường cong Z*(B)", markers=True)
            fig3.update_traces(line_color=LINE_COLOR)
            html_parts.append(_fig_to_html(fig3))
            
            # 4. Scenario x3 >= 30
            sc = res['scenario_x3']
            sc_html = "<h3>4. Kịch bản: Ưu tiên nhân lực số (x₃ ≥ 30)</h3>"
            if sc['status'] == 'Optimal':
                sc_html += f"<p>Khả thi. Z* thay đổi: {res['Z']:.2f} &rarr; {sc['Z']:.2f} (Δ = {sc['Z'] - res['Z']:.2f}).</p>"
                sc_html += f"<p>Phân bổ mới: {sc['allocation']}</p>"
            else:
                sc_html += "<p>Bài toán <b>không còn khả thi</b> khi thêm ràng buộc x₃ ≥ 30.</p>"
            html_parts.append(sc_html)
            
            ctx['graph_html'] = "".join(f"<div style='margin-bottom:30px;'>{part}</div>" for part in html_parts)
            
        ctx.update({'lp_status': res['status'], 'lp_z': f"{res['Z']:.2f}", 'total_budget_b2': tb, 'min_I': mi, 'min_AI': mai})

    elif current_bai == 3:
        w_growth = float(request.GET.get('w_growth', 0.15))
        w_productivity = float(request.GET.get('w_productivity', 0.15))
        w_spillover = float(request.GET.get('w_spillover', 0.20))
        w_export = float(request.GET.get('w_export', 0.15))
        w_employment = float(request.GET.get('w_employment', 0.10))
        w_ai = float(request.GET.get('w_ai', 0.20))
        w_risk = float(request.GET.get('w_risk', 0.15))
        res = solve_bai03(DATA_DIR, w_growth, w_productivity, w_spillover, w_export, w_employment, w_ai, w_risk)
        
        html_parts = []
        import pandas as pd
        
        # 1. Norm matrix
        df_norm = pd.DataFrame(res['norm_matrix'], columns=res['col_names'], index=res['sectors'])
        norm_html = f"<h3>1. Ma trận chuẩn hóa Min-Max (đảo dấu Rủi ro)</h3><div style='overflow-x:auto;'>{df_norm.to_html(float_format='%.4f', border=1)}</div>"
        html_parts.append(norm_html)
        
        # 2. Xếp hạng
        names = [r['sector_name_vi'] for r in res['ranking']]
        scores = [r['Priority'] for r in res['ranking']]
        fig_rank = px.bar(x=names, y=scores, title="2. Chỉ số ưu tiên theo ngành (giảm dần)", color=scores, color_continuous_scale='Blues')
        html_parts.append(_fig_to_html(fig_rank))
        
        df_rank = pd.DataFrame(res['ranking'])[['rank', 'sector_name_vi', 'Priority']]
        df_rank.columns = ['Hạng', 'Ngành', 'Priority']
        rank_html = f"<div style='overflow-x:auto;'>{df_rank.to_html(index=False, border=1)}</div>"
        html_parts.append(rank_html)
        
        # 3. Heatmap
        df_heatmap = pd.DataFrame(res['heatmap_data'], columns=[f"a₆={v}" for v in res['a6_values']], index=res['sectors'])
        fig_hm = px.imshow(df_heatmap, title="3. Phân tích độ nhạy: Heatmap Priority theo a₆ (AI Readiness)", labels=dict(x="Trọng số a₆", y="Ngành", color="Priority"), color_continuous_scale="Blues", aspect="auto")
        html_parts.append(_fig_to_html(fig_hm))
        top3_text = " | ".join([f"a₆={k}: {', '.join(v)}" for k, v in res['sensitivity_top3'].items()])
        html_parts.append(f"<p><b>Top-3 theo từng a₆:</b> {top3_text}</p>")
        
        # 4. So sánh
        sc = res['scenario_comparison']
        fig_g = px.bar(x=list(sc['growth']['scores'].keys()), y=list(sc['growth']['scores'].values()), title="4a. Điểm ưu tiên – Tăng trưởng", color=list(sc['growth']['scores'].values()), color_continuous_scale='Oranges')
        html_parts.append(f"<div><p><b>🚀 Định hướng Tăng trưởng - Top 3:</b> {', '.join(sc['growth']['top3'])}</p>{_fig_to_html(fig_g)}</div>")
        
        fig_i = px.bar(x=list(sc['inclusive']['scores'].keys()), y=list(sc['inclusive']['scores'].values()), title="4b. Điểm ưu tiên – Bao trùm", color=list(sc['inclusive']['scores'].values()), color_continuous_scale='Greens')
        html_parts.append(f"<div><p><b>🤝 Định hướng Bao trùm - Top 3:</b> {', '.join(sc['inclusive']['top3'])}</p>{_fig_to_html(fig_i)}</div>")
        
        ctx.update({'graph_html': "".join(f"<div style='margin-bottom:30px;'>{part}</div>" for part in html_parts), 'w_growth': w_growth, 'w_productivity': w_productivity, 'w_spillover': w_spillover, 'w_export': w_export, 'w_employment': w_employment, 'w_ai': w_ai, 'w_risk': w_risk})

    elif current_bai == 4:
        budget_b4 = int(request.GET.get('budget', 50000))
        w_gdp = float(request.GET.get('w_gdp', 0.40))
        w_equity = float(request.GET.get('w_equity', 0.25))
        w_ai = float(request.GET.get('w_ai', 0.20))
        res = solve_bai04(budget_b4, w_gdp=w_gdp, w_equity=w_equity, w_ai=w_ai)
        
        html_parts = []
        if res['status'] == 'Optimal':
            import pandas as pd
            # 1. PuLP
            df_pulp = pd.DataFrame(res['allocation']).T
            pulp_html = f"<h3>1. Giải bằng PuLP (CBC) – Ma trận phân bổ 6×4</h3><p style='color: green; font-weight: bold;'>Z* (PuLP) = {res['Z']:,.1f}</p><div style='overflow-x:auto;'>{df_pulp.to_html(float_format='%.1f', border=1)}</div>"
            html_parts.append(pulp_html)
            
            # 2. CVXPY
            cvxpy_html = "<h3>2. Giải bằng CVXPY – So sánh kết quả</h3>"
            if res['cvxpy_ok']:
                cvxpy_html += f"<p style='color: green; font-weight: bold;'>Z* (CVXPY) = {res['cvxpy_z']:,.1f}</p>"
                df_cvxpy = pd.DataFrame(res['cvxpy_alloc']).T
                cvxpy_html += f"<div style='overflow-x:auto; margin-bottom:10px;'>{df_cvxpy.to_html(float_format='%.1f', border=1)}</div>"
                diff_z = abs(res['Z'] - res['cvxpy_z'])
                if diff_z < 1.0:
                    cvxpy_html += f"<p style='color: #011F82; font-weight: bold;'>✅ Hai phương pháp cho kết quả giống nhau (chênh lệch Z*: {diff_z:,.2f} ≈ 0).</p>"
                else:
                    cvxpy_html += f"<p style='color: #D69E2E; font-weight: bold;'>⚠️ Chênh lệch Z*: {diff_z:,.2f}. Sự khác biệt nhỏ do solver SCS (CVXPY) là solver xấp xỉ.</p>"
            else:
                cvxpy_html += "<p style='color: #E53E3E;'>CVXPY không khả dụng hoặc không tìm được nghiệm tối ưu.</p>"
            html_parts.append(cvxpy_html)
            
            # 3. Heatmap
            df_hm = pd.DataFrame(res['allocation']).T
            fig_hm = px.imshow(df_hm, labels=dict(x="Hạng mục", y="Vùng", color="Ngân sách"), title="3. Heatmap Phân bổ Ngân sách theo Vùng và Hạng mục", color_continuous_scale="Blues", aspect="auto", text_auto=",.0f")
            row_sums = df_hm.sum(axis=1)
            max_region = row_sums.idxmax()
            hm_html = f"<div>{_fig_to_html(fig_hm)}<p><b>Nhận xét:</b> Vùng nhận ngân sách nhiều nhất: <b>{max_region}</b> ({row_sums[max_region]:,.0f} tỷ VND).</p></div>"
            html_parts.append(hm_html)
            
            # 4. Chi phí công bằng
            eq_html = "<h3>4. Chi phí kinh tế của công bằng vùng miền (bỏ C5)</h3>"
            eq_html += f"<ul style='margin-bottom:10px;'><li><b>Z* có C5 (Công bằng):</b> {res['Z']:,.1f}</li><li><b>Z* không C5:</b> {res['no_equity_z']:,.1f}</li><li><b>Chi phí công bằng (ΔZ):</b> {res['equity_cost']:,.1f} tỷ VND</li></ul>"
            if res['equity_cost'] > 0:
                eq_html += f"<p style='color: #E53E3E;'>Ràng buộc công bằng vùng miền (C5) làm giảm GDP gain <b>{res['equity_cost']:,.1f} tỷ VND</b>. Đây là 'chi phí' kinh tế mà xã hội trả để đảm bảo phát triển đồng đều giữa các vùng.</p>"
            else:
                eq_html += "<p style='color: #011F82;'>Ràng buộc công bằng không ảnh hưởng đến Z* (không có chi phí công bằng).</p>"
            
            df_noeq = pd.DataFrame(res['no_equity_alloc']).T
            fig_noeq = px.imshow(df_noeq, labels=dict(x="Hạng mục", y="Vùng", color="Ngân sách"), title="Heatmap KHÔNG có ràng buộc công bằng (bỏ C5)", color_continuous_scale="Reds", aspect="auto", text_auto=",.0f")
            eq_html += _fig_to_html(fig_noeq)
            html_parts.append(eq_html)
            
            ctx['graph_html'] = "".join(f"<div style='margin-bottom:30px;'>{part}</div>" for part in html_parts)
            
        else:
            ctx['graph_html'] = "<p style='color:red;'>Bài toán không khả thi.</p>"
            
        ctx.update({'budget_b4': budget_b4, 'w_gdp': w_gdp, 'w_equity': w_equity, 'w_ai': w_ai})
    elif current_bai == 5:
        b5 = int(request.GET.get('budget_b5', 80000))
        w_gdp = float(request.GET.get('w_gdp', 0.40))
        w_equity = float(request.GET.get('w_equity', 0.30))
        w_ai = float(request.GET.get('w_ai', 0.30))
        res = solve_bai05(b5, w_gdp=w_gdp, w_equity=w_equity, w_ai=w_ai)
        
        html_parts = []
        
        # 1. Base
        base_html = f"<h3>1. Kết quả giải gốc (PuLP - CBC)</h3><ul><li><b>Tổng Lợi ích (Z*):</b> {res['Z']:,.2f}</li><li><b>Tổng Chi phí:</b> {res['cost']:,.2f}</li><li><b>Tổng NPV:</b> {res['total_npv']:,.2f}</li><li><b>NPV Biên (Z*/Cost):</b> {res['npv_margin']:.4f}</li></ul>"
        if res['selected']:
            import pandas as pd
            df_proj = pd.DataFrame(res['projects']).T
            base_html += f"<div style='overflow-x:auto;'>{df_proj.to_html(float_format='%.2f', border=1)}</div>"
        else:
            base_html += "<p style='color:red;'>Infeasible - Không có dự án nào được chọn!</p>"
        html_parts.append(base_html)
        
        # 2. 100k
        r100 = res['res_100k']
        html_parts.append(f"<h3>2. Phân tích: Nới ngân sách lên 100.000 tỷ</h3><p><b>Lợi ích Z* (100k):</b> {r100['Z']:,.2f} (Δ = {r100['Z'] - res['Z']:,.2f})</p><p><b>Các dự án được chọn:</b> {', '.join(r100['selected'])}</p>")
        
        # 3. P1+P2
        rP = res['res_p1p2']
        p_html = "<h3>3. Phân tích: Bắt buộc chọn P1 và P2 (Redundancy)</h3>"
        if rP['status'] == 'Optimal':
            p_html += f"<p><b>Lợi ích Z* (P1+P2):</b> {rP['Z']:,.2f} (Δ = {rP['Z'] - res['Z']:,.2f})</p><p><b>Các dự án được chọn:</b> {', '.join(rP['selected'])}</p>"
        else:
            p_html += "<p style='color:red;'>❌ Không khả thi (Infeasible) do vi phạm ràng buộc ngân sách hoặc ràng buộc loại trừ C3.</p>"
        html_parts.append(p_html)
        
        # 4. Risk
        rR = res['res_risk']
        r_html = f"<h3>4. Mở rộng: Rủi ro dự án (Tối đa hóa lợi ích kỳ vọng E[Z])</h3><p><b>E[Z] Kỳ vọng:</b> {rR['Z']:,.2f}</p><p><b>Tập dự án chọn an toàn:</b> {', '.join(rR['selected'])}</p>"
        r_html += f"<details><summary>Bảng xác suất hoàn thành đúng tiến độ</summary><pre>{str(res['prob_completion'])}</pre></details>"
        html_parts.append(r_html)
        
        ctx.update({
            'graph_html': "".join(f"<div style='margin-bottom:30px;'>{part}</div>" for part in html_parts),
            'budget_b5': b5, 'w_gdp': w_gdp, 'w_equity': w_equity, 'w_ai': w_ai
        })

    elif current_bai == 6:
        weight_mode = int(request.GET.get('weight_mode', 0))
        w_manual_str = request.GET.get('w_manual', None)
        w_manual = [float(x) for x in w_manual_str.split(',')] if w_manual_str else None
        res = solve_bai06(DATA_DIR, w_manual=w_manual, weight_mode=weight_mode)
        
        html_parts = []
        import pandas as pd
        
        # 1. TOPSIS bar chart
        names = [r['region_name_vi'] for r in res['ranking']]
        scores = [r['TOPSIS'] for r in res['ranking']]
        mode_text = "Chuyên gia (Manual)" if weight_mode == 1 else "Entropy (Tự động)"
        fig_rank = px.bar(x=names, y=scores, title=f"1 & 2. Điểm TOPSIS (Trọng số: {mode_text})", color=scores, color_continuous_scale='Blues')
        html_parts.append(_fig_to_html(fig_rank))
        
        # 2. Heatmap sensitivity
        sens_data = []
        for w_val, ranks in res['sensitivity'].items():
            for region, score in ranks.items():
                sens_data.append({"w_AI": w_val, "Region": region, "Score": score})
        df_sens = pd.DataFrame(sens_data)
        df_pivot = df_sens.pivot(index="Region", columns="w_AI", values="Score")
        fig_heat = px.imshow(df_pivot, text_auto=".3f", aspect="auto", title="3. Biến động điểm TOPSIS khi thay đổi trọng số AI", labels=dict(x="Trọng số w_AI", y="Vùng", color="Score"), color_continuous_scale="Reds")
        html_parts.append(_fig_to_html(fig_heat))
        
        ctx.update({'graph_html': "".join(f"<div style='margin-bottom:30px;'>{part}</div>" for part in html_parts), 'weight_mode': weight_mode, 'w_manual': w_manual_str if w_manual_str else ""})

    elif current_bai == 7:
        n_gen = int(request.GET.get('n_gen', 50))
        pop_size = int(request.GET.get('pop_size', 50))
        res = solve_bai07(n_gen=n_gen, pop_size=pop_size)
        
        html_parts = []
        if res['n_pareto'] == 0:
            html_parts.append("<p style='color:red;'>Không tìm thấy nghiệm Pareto hợp lệ. Vui lòng kiểm tra lại solver hoặc thông số.</p>")
        else:
            # 1. 3D Scatter
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=res['f1_gdp'], y=res['f2_equity'], z=res['f3_env'],
                mode='markers', marker=dict(size=4, color=res['f1_gdp'], colorscale='Blues', opacity=0.8)
            )])
            fig_3d.update_layout(title='1. Pareto Front 3D (GDP, Equity, Emission)', scene=dict(xaxis_title='GDP (f1)', yaxis_title='Equity_MAD (f2)', zaxis_title='Emission (f3)'), margin=dict(l=0, r=0, b=0, t=40))
            html_parts.append(_fig_to_html(fig_3d))
            
            # 2. Parallel Coordinates
            fig_parc = go.Figure(data=
                go.Parcoords(
                    line = dict(color = res['f1_gdp'], colorscale = 'Viridis', showscale = True),
                    dimensions = list([
                        dict(range = [min(res['f1_gdp']), max(res['f1_gdp'])], label = 'GDP (Max)', values = res['f1_gdp']),
                        dict(range = [max(res['f2_equity']), min(res['f2_equity'])], label = 'Equity_MAD (Min)', values = res['f2_equity']),
                        dict(range = [max(res['f3_env']), min(res['f3_env'])], label = 'Emission (Min)', values = res['f3_env']),
                        dict(range = [max(res['f4_sec']), min(res['f4_sec'])], label = 'Security (Min)', values = res['f4_sec'])
                    ])
                )
            )
            fig_parc.update_layout(title="2. Biểu đồ Tọa độ song song (Parallel Coordinates) - 4 Mục tiêu", margin=dict(l=40, r=40, b=40, t=60))
            html_parts.append(_fig_to_html(fig_parc))
            
            # 3. TOPSIS Compromise
            top = res['topsis_compromise']
            html_parts.append(f"<h3>3. Nghiệm thỏa hiệp duy nhất (TOPSIS)</h3><ul><li><b>GDP:</b> {top['GDP']:,.2f}</li><li><b>Equity_MAD:</b> {top['Equity_MAD']:,.2f}</li><li><b>Emission:</b> {top['Emission']:,.2f}</li><li><b>Security:</b> {top['Security']:,.2f}</li></ul><p><i>*(Chọn từ {res['n_pareto']} nghiệm với trọng số: 0.40 GDP, 0.25 Bao trùm, 0.20 Môi trường, 0.15 An ninh)*</i></p>")
            
            # 4. Opportunity Cost
            opp = res['opportunity_cost']['best_gdp_sol']
            html_parts.append(f"<h3>4. Phân tích Chi phí cơ hội (Opportunity Cost)</h3><div style='background:#e6fffa; padding:10px; border-left: 4px solid #319795;'><b>Nghiệm Tăng trưởng cao nhất:</b><br>- GDP: {opp['GDP']:,.2f}<br>- Bao trùm: {opp['Equity_MAD']:,.2f}<br>- Môi trường: {opp['Emission']:,.2f}<br>- An ninh: {opp['Security']:,.2f}</div>")
            
        ctx.update({'graph_html': "".join(f"<div style='margin-bottom:30px;'>{part}</div>" for part in html_parts), 'n_gen': n_gen, 'pop_size': pop_size})

    elif current_bai == 8:
        discount = float(request.GET.get('discount', 0.05))
        capital_growth = float(request.GET.get('capital_growth', 0.06))
        target_ai = float(request.GET.get('target_ai', 0.85))
        budget_growth = float(request.GET.get('budget_growth', 0.08))
        res = solve_bai08(discount=discount, capital_growth=capital_growth, target_ai=target_ai, budget_growth=budget_growth)
        
        html_parts = []
        
        # 1. SLSQP
        html_parts.append(f"<h3>1. Giải bằng mô hình tối ưu động</h3><div style='background:#ebf8ff; padding:10px; border-left: 4px solid #3182CE;'><b>Tổng phúc lợi tối ưu (Z*):</b> {res['welfare_opt']:,.1f} tỷ VND</div>")
        
        # 2. Trajectories
        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(x=res['years'], y=res['K'], name='Vốn (K)', line=dict(color='#3182CE')))
        fig_traj.add_trace(go.Scatter(x=res['years'], y=res['Y'], name='Sản lượng (Y)', line=dict(color='#E53E3E')))
        fig_traj.add_trace(go.Scatter(x=res['years'], y=res['C'], name='Tiêu dùng (C)', line=dict(color='#38A169')))
        fig_traj.add_trace(go.Scatter(x=res['years'], y=res['D'], name='Số hóa (D)', yaxis='y2', line=dict(color='#D69E2E')))
        fig_traj.add_trace(go.Scatter(x=res['years'], y=res['H'], name='Nhân lực (H)', yaxis='y2', line=dict(color='#805AD5')))
        fig_traj.add_trace(go.Scatter(x=res['years'], y=res['AI'], name='AI', yaxis='y2', line=dict(color='#D53F8C', dash='dash')))
        fig_traj.update_layout(title='2. Quỹ đạo tối ưu các biến số vĩ mô (2026-2035)', yaxis2=dict(overlaying='y', side='right', title="Chỉ số phụ"))
        html_parts.append(_fig_to_html(fig_traj))
        
        # 3. Shock 2028
        fig_shock = go.Figure()
        fig_shock.add_trace(go.Scatter(x=res['years'], y=res['Y'], name='Y (Bình thường)', line=dict(color='#3182CE')))
        fig_shock.add_trace(go.Scatter(x=res['years'], y=res['Y_shock'], name='Y (Cú sốc)', line=dict(color='#E53E3E', dash='dash')))
        fig_shock.update_layout(title='3. So sánh quỹ đạo Sản lượng (Y) khi có cú sốc 2028')
        html_parts.append(f"<div><p><b>Welfare (Không cú sốc):</b> {res['welfare_opt']:,.1f} | <b>Welfare (Có cú sốc):</b> {res['welfare_shock']:,.1f} (Δ = {res['welfare_shock'] - res['welfare_opt']:,.1f})</p>{_fig_to_html(fig_shock)}</div>")
        
        # 4. Strategy Compare
        html_parts.append(f"<h3>4. So sánh chiến lược đầu tư: Trải đều vs Front-load</h3><ul><li><b>Welfare (Trải đều):</b> {res['welfare_even']:,.1f}</li><li><b>Welfare (Front-load):</b> {res['welfare_front']:,.1f}</li></ul><p style='color:#38A169; font-weight:bold;'>Chiến lược tốt hơn: {res['better_strategy']}</p>")
        
        ctx.update({'graph_html': "".join(f"<div style='margin-bottom:30px;'>{part}</div>" for part in html_parts), 'discount': discount, 'capital_growth': capital_growth, 'target_ai': target_ai, 'budget_growth': budget_growth})

    elif current_bai == 9:
        ai_rate = float(request.GET.get('ai_adoption_rate', 0.30))
        retrain = float(request.GET.get('retraining_budget', 15.0))
        speed = float(request.GET.get('transition_speed', 0.5))
        new_job = float(request.GET.get('new_job_multiplier', 0.4))
        res = solve_bai09(DATA_DIR, ai_adoption_rate=ai_rate, retraining_budget=retrain, transition_speed=speed, new_job_multiplier=new_job)
        
        html_parts = []
        import pandas as pd
        
        # 1. PuLP Net Job Table
        html_parts.append(f"<h3>1. Phân bổ tối ưu (PuLP)</h3><div style='font-size:18px; font-weight:bold; color:#E53E3E; margin-bottom:10px;'>Tổng Việc làm ròng (NetJob): {res['total_net']:,.2f} triệu</div>")
        df_sectors = pd.DataFrame(res['sector_table']).T
        html_parts.append(f"<div style='overflow-x:auto;'>{df_sectors.to_html(float_format='%.3f', border=1)}</div>")
        
        # 2. Threshold xH2
        html_parts.append(f"<h3>2. Ngưỡng đầu tư đào tạo ngành Chế biến chế tạo (Ngành 2)</h3><div style='background:#ebf8ff; padding:10px; border-left: 4px solid #3182CE;'><b>Ngưỡng $x_{{H, 2}}$ tối thiểu</b> để NetJob_2 $\ge$ 0 khi tối đa hóa AI ($x_{{AI, 2}}=1$): <b>{res['threshold_xH2']:.4f}</b></div>")
        
        # 3. Sankey
        fig_sankey = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15, thickness = 20, line = dict(color = "black", width = 0.5),
              label = res['sankey_nodes'], color = ["blue", "red", "green", "orange", "purple"]
            ),
            link = dict(
              source = res['sankey_links']['source'], target = res['sankey_links']['target'], value = res['sankey_links']['value']
            ))])
        fig_sankey.update_layout(title_text="3. Biểu đồ Sankey: Dịch chuyển lao động phổ thông", font_size=12)
        html_parts.append(_fig_to_html(fig_sankey))
        
        # 4. Extension constraint
        html_parts.append("<h3>4. Ràng buộc mở rộng: Không ngành nào mất quá 5% lao động</h3>")
        if res['ext_feasible']:
            html_parts.append("<div style='background:#f0fff4; padding:10px; border-left: 4px solid #38a169; color:#276749;'>✅ <b>Khả thi:</b> Mô hình CÓ THỂ tìm được phân bổ thỏa mãn điều kiện không ngành nào mất >5% lao động.</div>")
        else:
            html_parts.append("<div style='background:#fff5f5; padding:10px; border-left: 4px solid #e53e3e; color:#9b2c2c;'>❌ <b>Không khả thi:</b> Ngân sách đào tạo không đủ hoặc tốc độ chuyển đổi quá chậm để giữ mức mất việc <5% ở tất cả các ngành.</div>")
            
        ctx.update({'graph_html': "".join(f"<div style='margin-bottom:30px;'>{part}</div>" for part in html_parts), 'ai_adoption_rate': ai_rate, 'retraining_budget': retrain, 'transition_speed': speed, 'new_job_multiplier': new_job})

    elif current_bai == 10:
        p1 = float(request.GET.get('p_optimistic', 0.30))
        p2 = float(request.GET.get('p_baseline', 0.45))
        p3 = float(request.GET.get('p_pessimistic', 0.20))
        budget_b10 = float(request.GET.get('first_stage_cap', 65.0))
        res = solve_bai10(p_optimistic=p1, p_baseline=p2, p_pessimistic=p3, first_stage_cap=budget_b10)
        
        html_parts = []
        import pandas as pd
        
        # 1. EVPI / VSS
        html_parts.append(f"<h3>1. So sánh Lợi nhuận kỳ vọng</h3><ul><li><b>Giải pháp ngẫu nhiên (SP):</b> {res['sp_value']:,.1f}</li><li><b>Kỳ vọng giá trị hoàn hảo (EVPI):</b> {res['evpi']:,.1f} <i>(Lợi ích thêm nếu biết trước tương lai)</i></li><li><b>Giá trị giải pháp ngẫu nhiên (VSS):</b> {res['vss']:,.1f} <i>(Tổn thất nếu chỉ dùng trung bình EEV)</i></li></ul>")
        
        # 2. Allocation Table
        df_alloc = pd.DataFrame(res['sp_alloc']).T
        html_parts.append(f"<h3>2. Quyết định phân bổ Giai đoạn 1 (First-stage)</h3><div style='overflow-x:auto;'>{df_alloc.to_html(float_format='%.2f', border=1)}</div>")
        
        # 3. Robust Optimization
        html_parts.append(f"<h3>3. Robust Optimization (Cực đại hóa kịch bản xấu nhất)</h3><div style='background:#ebf8ff; padding:10px; border-left: 4px solid #3182CE;'>Giá trị lợi ích tồi tệ nhất được đảm bảo (Worst-case Z): <b>{res['rob_value']:,.1f}</b></div>")
        
        categories = ['I (Hạ tầng)', 'D (Số hóa)', 'AI', 'H (Nhân lực)']
        fig_sp = px.pie(names=categories, values=res['x_sp'], title="Phân bổ SP (Tối đa hóa kỳ vọng)")
        fig_rob = px.pie(names=categories, values=res['x_rob'], title="Phân bổ Robust (An toàn nhất)")
        
        html_parts.append(f"<div style='display:flex; flex-wrap:wrap; gap:20px;'><div style='flex:1; min-width:300px;'>{_fig_to_html(fig_sp)}</div><div style='flex:1; min-width:300px;'>{_fig_to_html(fig_rob)}</div></div>")
        
        ctx.update({'graph_html': "".join(f"<div style='margin-bottom:30px;'>{part}</div>" for part in html_parts), 'p_optimistic': p1, 'p_baseline': p2, 'p_pessimistic': p3, 'first_stage_cap': budget_b10})

    elif current_bai == 11:
        alpha = float(request.GET.get('alpha', 0.1))
        gamma = float(request.GET.get('gamma', 0.95))
        episodes = int(request.GET.get('episodes', 10000))
        use_dqn = request.GET.get('use_dqn', 'true') == 'true'
        res = solve_bai11(learning_rate=alpha, discount_factor=gamma, episodes=episodes, use_dqn=use_dqn)
        
        html_parts = []
        import pandas as pd
        
        # 1. Learning Curve
        fig_lc = go.Figure()
        x_q = list(range(0, res['episodes'], max(1, res['episodes']//100)))
        fig_lc.add_trace(go.Scatter(x=x_q, y=res['q_smoothed'], mode='lines', name='Q-Learning (Tabular)', line=dict(color='#3182CE')))
        if len(res['dqn_smoothed']) > 0:
            x_dqn = list(range(0, res['episodes'], max(1, res['episodes']//100)))
            x_dqn = x_dqn[:len(res['dqn_smoothed'])]
            fig_lc.add_trace(go.Scatter(x=x_dqn, y=res['dqn_smoothed'], mode='lines', name='DQN (Neural Net)', line=dict(color='#E53E3E')))
        fig_lc.update_layout(title="1. Đánh giá Learning Curve (Tổng phần thưởng trung bình)", xaxis_title="Episodes", yaxis_title="Reward")
        html_parts.append(_fig_to_html(fig_lc))
        
        # 2. Optimal Policy
        html_parts.append("<h3>2. Chính sách tối ưu \pi^*(s) tại các trạng thái khởi đầu</h3>")
        df_policies = pd.DataFrame(list(res['extracted_policies'].items()), columns=["Trạng thái giả định", "Hành động (Policy) được chọn"])
        html_parts.append(f"<div style='overflow-x:auto;'>{df_policies.to_html(index=False, border=1)}</div>")
        
        # 3. Rules Performance
        df_rules = pd.DataFrame(list(res['rules_perf'].items()), columns=["Chính sách", "Phần thưởng trung bình"])
        fig_bar = px.bar(df_rules, x="Chính sách", y="Phần thưởng trung bình", color="Chính sách", title="3. So sánh hiệu suất với Rule-based Policies")
        html_parts.append(_fig_to_html(fig_bar))
        
        ctx.update({'graph_html': "".join(f"<div style='margin-bottom:30px;'>{part}</div>" for part in html_parts), 'alpha': alpha, 'gamma': gamma, 'episodes': episodes, 'use_dqn': use_dqn})

    elif current_bai == 12:
        budget = float(request.GET.get('budget', 50000.0))
        scenario = request.GET.get('scenario', 'S5')
        res = solve_bai12_dashboard(DATA_DIR, total_budget=budget, scenario=scenario)
        
        html_parts = []
        
        html_parts.append(f"<div style='background:#ebf8ff; padding:15px; border-left: 5px solid #3182CE; margin-bottom:20px; font-size:16px;'><b>Mô tả kịch bản:</b> {res['description']}</div>")
        
        html_parts.append("<h3>1. Phân bổ ngân sách (Tỷ VND)</h3>")
        alloc = res['allocation']
        html_parts.append(f"<div style='display:flex; justify-content:space-around; background:#f7fafc; padding:15px; border-radius:8px;'><div><b>Hạ tầng (I):</b> {alloc['I']:,.0f}</div><div><b>Số hóa (D):</b> {alloc['D']:,.0f}</div><div><b>AI:</b> {alloc['AI']:,.0f}</div><div><b>Nhân lực (H):</b> {alloc['H']:,.0f}</div></div>")
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=res['radar']['values'], theta=res['radar']['dimensions'], fill='toself', line_color=LINE_COLOR
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Đánh giá tổng hợp đa chiều")
        html_parts.append(_fig_to_html(fig_radar))
        
        # 2. Comparison S1, S3, S5
        html_parts.append("<h3>2. So sánh 3 kịch bản chính (Năm 2030)</h3>")
        s1 = solve_bai12_dashboard(DATA_DIR, budget, 'S1')
        s3 = solve_bai12_dashboard(DATA_DIR, budget, 'S3')
        s5 = solve_bai12_dashboard(DATA_DIR, budget, 'S5')
        
        import pandas as pd
        comp_data = {
            "Chỉ số (2030)": ["GDP (Nghìn tỷ VND)", "Tỷ trọng AI (%)", "Điểm Rủi ro", "Mất việc ròng (Ngàn người)"],
            "S1 (Truyền thống)": [
                f"{s1['gdp_forecast']['gdp'][-1]:,.2f}", f"{s1['risk']['ai_budget_share']}%", f"{s1['risk']['risk_score']} ({s1['risk']['level']})", f"{s1['labor_impact']['net_total']:,.1f}"
            ],
            "S3 (AI dẫn dắt)": [
                f"{s3['gdp_forecast']['gdp'][-1]:,.2f}", f"{s3['risk']['ai_budget_share']}%", f"{s3['risk']['risk_score']} ({s3['risk']['level']})", f"{s3['labor_impact']['net_total']:,.1f}"
            ],
            "S5 (Tối ưu cân bằng)": [
                f"{s5['gdp_forecast']['gdp'][-1]:,.2f}", f"{s5['risk']['ai_budget_share']}%", f"{s5['risk']['risk_score']} ({s5['risk']['level']})", f"{s5['labor_impact']['net_total']:,.1f}"
            ]
        }
        df_comp = pd.DataFrame(comp_data)
        html_parts.append(f"<div style='overflow-x:auto;'>{df_comp.to_html(index=False, border=1)}</div>")
        
        # 3. GDP and Labor
        html_parts.append("<h3>3. Tăng trưởng GDP & Việc làm</h3>")
        fig_gdp = px.line(x=res['gdp_forecast']['years'], y=res['gdp_forecast']['gdp'], title="Dự báo GDP 2026-2030 (Nghìn tỷ VND)", markers=True)
        fig_gdp.update_traces(line_color=LINE_COLOR)
        html_parts.append(_fig_to_html(fig_gdp))
        
        fig_labor = px.bar(x=res['labor_impact']['sectors'], y=res['labor_impact']['net_jobs'], title="Tác động việc làm ròng", color=res['labor_impact']['net_jobs'], color_continuous_scale='Blues')
        html_parts.append(_fig_to_html(fig_labor))
        
        # 4. Risk and Topsis
        html_parts.append("<h3>4. Rủi ro & Vùng miền</h3>")
        df_topsis = pd.DataFrame(res['topsis'])
        fig_topsis = px.bar(df_topsis, x='region', y='score', title="Điểm TOPSIS các vùng", text='rank', color='score', color_continuous_scale='Blues')
        html_parts.append(_fig_to_html(fig_topsis))
        
        ctx.update({'graph_html': "".join(f"<div style='margin-bottom:30px;'>{part}</div>" for part in html_parts), 'budget': budget, 'scenario': scenario})

    return render(request, 'dashboard/index.html', ctx)
