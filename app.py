"""Bearing Health Monitor - multi-page Streamlit app.  Run: streamlit run app.py"""
import os, time
import numpy as np, pandas as pd, streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import kurtosis
from scipy.signal import butter, filtfilt, hilbert

st.set_page_config(page_title='Bearing Health Monitor', layout='wide')
HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = ['NORMAL', 'WATCH', 'DERATE', 'SAFE_STOP']
COLORS = {'NORMAL': '#2e7d32', 'WATCH': '#f9a825', 'DERATE': '#ef6c00', 'SAFE_STOP': '#c62828'}
ACTIONS = {'NORMAL': 'None. Keep running, check every 10 minutes',
           'WATCH': 'Inspect at next planned stop, check lubrication and alignment',
           'DERATE': 'Cut speed or load by 30%, book bearing replacement within 48 h',
           'SAFE_STOP': 'Stop the machine, replace the bearing before restart (manual reset)'}
PRESETS = {'CWRU drive-end bearing (SKF 6205)': dict(n=9, d=7.94, D=39.04, ang=0.0, rpm=1797, fs=12000),
           'IMS bearing (Rexnord ZA-2115)': dict(n=16, d=8.407, D=71.501, ang=15.17, rpm=2000, fs=20000)}
NAMES = {'BPFO': 'Outer race', 'BPFI': 'Inner race', 'BSF': 'Ball'}

st.markdown("""<style>
.block-container{padding-top:1rem;max-width:1250px}
.hero{background:linear-gradient(120deg,#0b2545,#13408a);padding:20px 28px;border-radius:14px;margin-bottom:14px}
.hero h1,.hero p,.hero b{color:#fff!important;margin:0}.hero h1{font-size:29px}.hero p{font-size:14px;opacity:.9;margin-top:4px}
.card{background:#f2f5fb;border:1px solid #dbe3f1;border-radius:12px;padding:14px 16px;min-height:112px}
.card,.card *{color:#1b2a41}.kpi{font-size:30px;font-weight:700;color:#13408a!important}.lbl{font-size:13px}
.badge{padding:14px;border-radius:10px;color:#fff;font-size:26px;font-weight:700;text-align:center}
</style>""", unsafe_allow_html=True)


def hero(sub):
    st.markdown(f"<div class='hero'><h1>Bearing Health Monitor</h1>"
                f"<p>ML based predictive maintenance system for industrial equipment | {sub}</p>"
                f"<p><b>Ishika Pandey, Khushi Khobar, Nikita Kumari</b> | Group 3, MAE-1, IGDTUW | "
                f"Guide: Prof. Arvind Jayant</p></div>", unsafe_allow_html=True)


def badge(text, color):
    st.markdown(f"<div class='badge' style='background:{color}'>{text}</div>", unsafe_allow_html=True)


def card(col, big, small):
    col.markdown(f"<div class='card'><div class='kpi'>{big}</div><div class='lbl'>{small}</div></div>",
                 unsafe_allow_html=True)


# ------------------------------------------------------------ data + logic
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(HERE, 'ims2_health.csv'))
    df = df[df['rms'] > 0.02].reset_index(drop=True)
    df['hours'] = np.arange(len(df)) * 10 / 60
    return df


def build_health(rms, monotone=False):
    sm = pd.Series(rms).rolling(5, center=True, min_periods=1).median().values
    base, fail = sm[:300].mean(), np.percentile(sm, 99)
    h = 100 * (1 - np.clip((sm - base) / (fail - base), 0, 1))
    return (np.minimum.accumulate(h) if monotone else h), sm


def onset_hour(sm, hours):
    above = sm > sm[:300].mean() + 5 * sm[:300].std()
    return next((hours[i] for i in range(len(above) - 5) if above[i:i + 5].all()), None)


def run_fsm(h, hours, tw, td, ts, n_good, hyst=True):
    level, good, ch, first, states = 0, 0, 0, {}, []
    for i, v in enumerate(h):
        if level == 3:
            states.append(LEVELS[3]); continue
        t = 3 if v < ts else 2 if v < td else 1 if v < tw else 0
        if t > level:
            level, good = t, 0; ch += 1; first.setdefault(t, hours[i])
        elif t < level:
            if hyst:
                good += 1
                if good >= n_good: level -= 1; good = 0; ch += 1
            else:
                level = t; ch += 1
        else:
            good = 0
        states.append(LEVELS[level])
    return states, ch, first


def est_rul(h, hours, i, w=50):
    lo = max(0, i - w + 1)
    if i - lo < 10: return None
    s = np.polyfit(hours[lo:i + 1], h[lo:i + 1], 1)[0]
    return None if s > -0.02 else max(h[i] / -s, 0.0)


def fault_freqs(P, fr):
    r = P['d'] / P['D'] * np.cos(np.radians(P['ang']))
    return {'BPFO': P['n'] / 2 * (1 - r) * fr, 'BPFI': P['n'] / 2 * (1 + r) * fr,
            'BSF': P['D'] / (2 * P['d']) * (1 - r ** 2) * fr}


def env_spec(x, fs):
    b, a = butter(4, [2000, min(5000, 0.45 * fs)], btype='band', fs=fs)
    e = np.abs(hilbert(filtfilt(b, a, x))); e -= e.mean()
    return np.fft.rfftfreq(len(e), 1 / fs), np.abs(np.fft.rfft(e)) / len(e)


def diagnose(freq, spec, F):
    m = (freq > 20) & (freq < 600)
    floor = np.median(spec[m]) + 1e-12
    sc = {}
    for k, f0 in F.items():
        v = [spec[(freq > h * f0 * .98) & (freq < h * f0 * 1.02)].max()
             for h in (1, 2, 3) if ((freq > h * f0 * .98) & (freq < h * f0 * 1.02)).any()]
        sc[k] = max(v) / floor if v else 0.0
    best = max(sc, key=sc.get)
    return (best if sc[best] >= 8 else None), sc


def sim_signal(fault, fr, F, fs, sev, snr, seed, T=2.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * T)) / fs
    x = 0.05 * np.sin(2 * np.pi * fr * t) + 0.02 * np.sin(4 * np.pi * fr * t)
    if fault != 'Healthy':
        f0 = {'Outer race': F['BPFO'], 'Inner race': F['BPFI'], 'Ball': 2 * F['BSF']}[fault]
        tt = np.arange(0, 0.02, 1 / fs)
        burst = np.exp(-tt / 0.0015) * np.sin(2 * np.pi * 3500 * tt)
        pos = 0.0
        while pos < T - 0.02:
            seg = x[int(pos * fs):int(pos * fs) + len(burst)]
            seg += sev * burst[:len(seg)]
            pos += (1 / f0) * (1 + 0.01 * rng.standard_normal())
        if fault == 'Inner race': x *= 1 + 0.5 * np.sin(2 * np.pi * fr * t)
    return t, x + rng.standard_normal(len(x)) * np.sqrt(np.mean(x ** 2) / 10 ** (snr / 10))


def analyse(x, fs, F, title):
    freq, spec = env_spec(x, fs)
    best, sc = diagnose(freq, spec, F)
    c = st.columns(4)
    with c[0]:
        badge(NAMES[best].upper() if best else 'NO CLEAR FAULT', '#c62828' if best else '#2e7d32')
    c[1].metric('RMS', f'{np.sqrt(np.mean(x ** 2)):.3f}')
    c[2].metric('Kurtosis', f'{kurtosis(x):.2f}')
    c[3].metric('Strongest match', f'{max(sc.values()):.1f}x noise floor')
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    n = int(0.3 * fs)
    ax[0].plot(np.arange(n) / fs, x[:n], lw=.7, color='#1f4e79'); ax[0].set_title(title + ': first 0.3 s')
    ax[0].set_xlabel('Time (s)')
    m = (freq > 5) & (freq < 600)
    ax[1].plot(freq[m], spec[m], color='#1f4e79', lw=1)
    for (k, f0), col in zip(F.items(), ['#c62828', '#2e7d32', '#ef6c00']):
        ax[1].axvline(f0, color=col, ls='--', lw=1.2, label=f'{NAMES[k]} theory {f0:.0f} Hz')
    ax[1].set_title('Envelope spectrum: peaks should land on the theory lines')
    ax[1].set_xlabel('Frequency (Hz)'); ax[1].legend(fontsize=8)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption('Scores (peak vs noise floor): ' + ', '.join(f'{NAMES[k]} {v:.1f}' for k, v in sc.items())
               + '. A match above 8 is called a fault.')


def live_fig(i, hrs, h, rms, states, tw, td, ts, onset):
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
    for lo, hi, k in [(tw, 102, 'NORMAL'), (td, tw, 'WATCH'), (ts, td, 'DERATE'), (0, ts, 'SAFE_STOP')]:
        ax[0].axhspan(lo, hi, color=COLORS[k], alpha=.13)
    ax[0].plot(hrs, h, color='#cccccc', lw=1); ax[0].plot(hrs[:i + 1], h[:i + 1], color='#1f4e79', lw=2)
    ax[0].scatter([hrs[i]], [h[i]], color=COLORS[states[i]], s=80, zorder=5, edgecolor='k')
    for k in range(1, i + 1):
        if states[k] != states[k - 1]:
            ax[0].axvline(hrs[k], color=COLORS[states[k]], ls='--', lw=1)
    if onset: ax[0].axvline(onset, color='grey', ls=':', lw=1)
    ax[0].set_ylim(0, 102); ax[0].set_ylabel('Health score'); ax[0].set_title('Health score and maintenance state')
    ax[1].plot(hrs, rms, color='#cccccc', lw=.8); ax[1].plot(hrs[:i + 1], rms[:i + 1], color='#1f4e79', lw=1.2)
    ax[1].set_ylabel('RMS'); ax[1].set_xlabel('Hours since start')
    fig.tight_layout(); return fig


# ------------------------------------------------------------------ pages
def page_home():
    hero('overview')
    c = st.columns(4)
    card(c[0], '74.8 h', 'warning window before failure on a real run-to-failure recording (NASA IMS)')
    card(c[1], '< 1 Hz', 'gap between measured and theoretical fault frequency (CWRU physics check)')
    card(c[2], '100%', 'accuracy on motor loads the model never saw (CWRU, clean data)')
    card(c[3], '80.5%', 'accuracy at -5 dB noise after noise-aware training (29.9% without)')
    st.subheader('What this project does, in plain words')
    st.write('Machines have bearings, and bearings crack. A cracked bearing shakes in a particular way. '
             'Our software reads that shaking, says which part is damaged, tracks how bad it is getting, and '
             'decides what maintenance to do, writing the work order automatically.')
    st.subheader('The system in 6 steps')
    s = [('1. Signal', 'Vibration recordings from public datasets (CWRU, NASA IMS)'),
         ('2. Features', 'Cut into windows, describe each with RMS, kurtosis and more'),
         ('3. Diagnose', 'Random Forest names the fault; physics check confirms it'),
         ('4. Health', 'A 0 to 100 health score tracks the bearing as it wears'),
         ('5. Decide', 'State machine: Normal, Watch, Derate, Safe Stop'),
         ('6. Act', 'Automatic work order and this dashboard')]
    for row in (s[:3], s[3:]):
        c = st.columns(3)
        for col, (a, b) in zip(c, row): card(col, a, b)
    st.subheader('How to use this dashboard')
    st.write('Use the menu on the left. **Live Monitor** replays a bearing from healthy to failure and lets you change '
             'the decision thresholds. **Simulator** lets you choose a fault, speed and noise level, then diagnoses it '
             'from physics, or upload a real recording. **Results** shows our test evidence. **About** lists limits.')


def page_live():
    hero('live monitor')
    df = load_data(); hours, rms = df['hours'].values, df['rms'].values
    n, end_h = len(df), df['hours'].iloc[-1]
    with st.expander('Decision settings: change them and watch the outputs change', expanded=True):
        a, b, c, d = st.columns(4)
        tw = a.slider('Watch below (health)', 60, 100, 98); td = b.slider('Derate below', 30, 99, 72)
        ts = c.slider('Safe Stop below', 5, 90, 50); ng = d.slider('Good readings to step back', 1, 10, 3)
        mono = st.checkbox('Health can only get worse')
    if not ts < td < tw:
        st.error('Order must be Safe Stop < Derate < Watch.'); return
    h, sm = build_health(rms, mono)
    states, _, _ = run_fsm(h, hours, tw, td, ts, ng)
    onset = onset_hour(sm, hours)
    c1, c2, c3 = st.columns([4, 1.3, 1.3])
    pos = c1.slider('Replay position (one file = 10 minutes)', 0, n - 1, 0)
    step = c2.selectbox('Play speed', [2, 5, 10, 20], index=2)
    play = c3.button('Play replay')
    live = st.empty()

    def render(i):
        s, r = states[i], est_rul(h, hours, i)
        k = st.columns([1.6, 1, 1, 1, 1])
        with k[0]: badge(s, COLORS[s])
        k[1].metric('Health', f'{h[i]:.0f} / 100'); k[2].metric('RMS', f'{rms[i]:.3f}')
        k[3].metric('Est. hours left', 'stable' if r is None else f'{min(r, 999):.0f} h')
        k[4].metric('Actual hours left', f'{end_h - hours[i]:.0f} h')
        fig = live_fig(i, hours, h, rms, states, tw, td, ts, onset); st.pyplot(fig); plt.close(fig)
        l, rr = st.columns(2)
        with l:
            st.subheader('Automatic work order')
            if s == 'NORMAL': st.success('No work order. Machine running normally.')
            else:
                st.table(pd.DataFrame({'Field': ['Machine', 'Fault', 'State', 'Time', 'Health', 'Action'],
                    'Value': ['Test rig, bearing 1', 'Outer race', s, f'hour {hours[i]:.1f}', f'{h[i]:.0f} / 100',
                              ACTIONS[s]]}).set_index('Field'))
        with rr:
            st.subheader('State changes so far')
            rows = [{'Hour': round(hours[k], 1), 'To': states[k], 'Health': round(h[k]),
                     'Hours before failure': round(end_h - hours[k], 1)}
                    for k in range(1, i + 1) if states[k] != states[k - 1]]
            st.dataframe(pd.DataFrame(rows), hide_index=True) if rows else st.write('None yet.')

    if play:
        for p in list(range(pos, n, step)) + [n - 1]:
            with live.container(): render(p)
            time.sleep(0.05)
    else:
        with live.container(): render(pos)
    st.caption(f'Thresholds are tuned on this one run (failure at {end_h:.1f} h). Estimated hours left is a '
               'straight-line fit and is only indicative.')


def page_sim():
    hero('simulator and diagnosis')
    st.write('Choose a bearing, a fault, a speed and a noise level. The app builds a vibration signal from bearing '
             'physics, then finds the fault from the envelope spectrum, with no training involved.')
    L, R = st.columns([1, 2.3])
    with L:
        pn = st.selectbox('Bearing', list(PRESETS)); P = PRESETS[pn]
        fault = st.selectbox('Fault to simulate', ['Outer race', 'Inner race', 'Ball', 'Healthy'])
        rpm = st.slider('Shaft speed (rpm)', 600, 3600, P['rpm'], 10)
        sev = st.slider('Fault severity', 0.1, 2.0, 0.8, 0.1)
        snr = st.slider('Noise: SNR in dB (lower = noisier)', -10, 30, 10)
        seed = st.number_input('Random seed', 0, 999, 1)
    F = fault_freqs(P, rpm / 60)
    with R:
        t, x = sim_signal(fault, rpm / 60, F, P['fs'], sev, snr, int(seed))
        analyse(x, P['fs'], F, 'Simulated signal')
    with st.expander('Try a real recording: CWRU .mat file or one-column CSV'):
        up = st.file_uploader('Upload file', type=['mat', 'csv', 'txt'])
        a, b = st.columns(2)
        fs_u = a.number_input('Sampling rate (Hz)', 1000, 100000, 12000)
        rpm_u = b.number_input('Shaft speed (rpm)', 300, 6000, 1797)
        if up is not None:
            try:
                if up.name.endswith('.mat'):
                    m = loadmat(up); key = [k for k in m if k.endswith('DE_time')]
                    xr = m[key[0]].ravel()
                else:
                    xr = pd.read_csv(up, header=None).iloc[:, 0].values.astype(float)
                analyse(xr[:int(fs_u * 5)], int(fs_u), fault_freqs(PRESETS['CWRU drive-end bearing (SKF 6205)'],
                        rpm_u / 60), 'Uploaded recording')
                st.caption('Uses the CWRU bearing geometry. Physics screening is a demo and can miss faults on real data.')
            except Exception as e:
                st.error(f'Could not read that file: {e}')


def page_results():
    hero('results')
    df = load_data(); h, sm = build_health(df['rms'].values); hours = df['hours'].values
    end_h = hours[-1]; onset = onset_hour(sm, hours)
    st.subheader('1. Diagnosis on CWRU, tested on motor loads never seen in training')
    a, b = st.columns(2)
    cm = pd.DataFrame(np.diag([192] * 4), index=['ball', 'inner', 'normal', 'outer'],
                      columns=['ball', 'inner', 'normal', 'outer'])
    a.write('Confusion matrix (rows = true, columns = predicted)'); a.dataframe(cm)
    b.info('All 768 test windows correct. This means the task is easy on clean lab data, so we also stress-tested it.')
    st.subheader('2. Noise stress test')
    nt = pd.DataFrame({'SNR (dB)': ['clean', '20', '10', '5', '0', '-5'],
                       'Trained on clean data': [100, 100, 100, 100, 95.7, 29.9],
                       'Trained with noise': [99.5, 99.5, 99.6, 100, 100, 80.5]}).set_index('SNR (dB)')
    a, b = st.columns(2); a.line_chart(nt); b.dataframe(nt)
    st.subheader('3. Physics check: measured vs theoretical fault frequency')
    st.dataframe(pd.DataFrame({'Fault': ['Outer race', 'Inner race'], 'Theory (Hz)': [107.4, 162.2],
                               'Measured peak (Hz)': [107.6, 161.6]}), hide_index=True)
    st.subheader('4. Degradation and decisions on the IMS run')
    s1, _, f1 = run_fsm(h, hours, 98, 72, 50, 3, True); _, c0, _ = run_fsm(h, hours, 98, 72, 50, 3, False)
    _, c1, _ = run_fsm(h, hours, 98, 72, 50, 3, True)
    rms = df['rms'].values; thr = rms[:300].mean() + 3 * rms[:300].std(); al = rms > thr
    k = st.columns(4)
    card(k[0], f'{onset:.1f} h', 'degradation onset'); card(k[1], f'{end_h:.1f} h', 'failure')
    card(k[2], f'{c0} vs {c1}', 'state changes: no hysteresis vs hysteresis')
    card(k[3], f'{hours[al][0]:.1f} h', 'first alarm of a plain RMS threshold')
    st.write(f'Our first Watch is at {f1[1]:.1f} h, Derate at {f1[2]:.1f} h ({end_h - f1[2]:.1f} h before failure) '
             f'and Safe Stop at {f1[3]:.1f} h ({end_h - f1[3]:.1f} h before failure).')
    st.warning('Honest reading: a plain threshold warns about as early as we do. Our value is graded decisions with '
               'work orders. Isolation Forest also matched it, and a straight-line remaining-life fit failed on this '
               'bearing. Everything is tuned on one run, so results are indicative.')
    shown = False
    for fn, cap in [('envelope_physics.png', 'Envelope spectrum vs theory (CWRU)'),
                    ('noise_test_compare.png', 'Noise test, before and after noise training'),
                    ('confusion_unseen_load.png', 'Confusion matrix, unseen loads'),
                    ('ims2_health_clean.png', 'IMS health curve')]:
        p = next((q for q in (os.path.join(HERE, 'assets', fn), os.path.join(HERE, fn)) if os.path.exists(q)), '')
        if p:
            st.image(p, caption=cap)
            shown = True
    if not shown: st.caption('Add your saved plots to an assets folder next to app.py and they appear here.')


def page_about():
    hero('about')
    st.subheader('Data')
    st.write('Both datasets are public benchmarks. We did not collect them, and no hardware was used. '
             '**CWRU** (Case Western Reserve University Bearing Data Center) has healthy and deliberately damaged '
             'bearings at four motor loads. **IMS** (NASA / University of Cincinnati) has bearings run until failure, '
             'with a 1-second recording every 10 minutes.')
    st.subheader('What we built ourselves')
    st.write('Loading and labelling, features, models, the unseen-load test design, the noise test, the physics check, '
             'the health score and onset rule, the state machine with hysteresis and latch, the work orders, and '
             'this dashboard.')
    st.subheader('Limits, stated plainly')
    st.markdown('- Lab data is cleaner than a factory.\n- One run-to-failure bearing, so thresholds and remaining-life '
                'estimates are indicative only.\n- The simulator is synthetic. Physics screening on uploaded files is a '
                'demo and can miss faults.\n- Edge deployment on a microcontroller and a real sensor are future work.')


PAGES = {'Overview': page_home, 'Live Monitor': page_live, 'Simulator': page_sim,
         'Results': page_results, 'About': page_about}
st.sidebar.title('Navigate')
PAGES[st.sidebar.radio('Page', list(PAGES), key='page', label_visibility='collapsed')]()
st.sidebar.caption('Group 3, MAE-1, IGDTUW')
