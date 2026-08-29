import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.6/full/pyodide.mjs";

let pyodide = null;
let ready = false;
let h5pyLoaded = false;

function send(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

async function initialize() {
  send("status", { message: "Loading Python/WebAssembly runtime…" });
  pyodide = await loadPyodide();
  send("status", { message: "Loading NumPy and SciPy…" });
  await pyodide.loadPackage(["numpy", "scipy"]);

  const coreUrl = new URL("./fibdisp_core.py", import.meta.url);
  const response = await fetch(coreUrl);
  if (!response.ok) throw new Error(`Cannot load fibdisp_core.py (${response.status})`);
  const coreSource = await response.text();
  pyodide.FS.writeFile("/home/pyodide/fibdisp_core.py", coreSource);

  pyodide.runPython(`
import sys
if "/home/pyodide" not in sys.path:
    sys.path.insert(0, "/home/pyodide")
import fibdisp_core as fc
import numpy as np
import json
_last_res = None
_last_sweep = None
_last_run_params = None
`);

  ready = true;
  send("ready", { pyodideVersion: pyodide.version });
}

const readyPromise = initialize().catch((error) => {
  send("error", { scope: "runtime", message: `Initialization failed: ${error.message}` });
  throw error;
});

function progressBridge(scope) {
  return (value) => send("progress", { scope, value: Number(value) });
}

function setInputs(params, extra = {}) {
  pyodide.globals.set("_params_json", JSON.stringify(params));
  for (const [key, value] of Object.entries(extra)) {
    if (typeof value === "object" && value !== null) pyodide.globals.set(key, JSON.stringify(value));
    else pyodide.globals.set(key, value);
  }
}

function parsePythonJson(source) {
  return JSON.parse(pyodide.runPython(source));
}

async function previewInput(params) {
  await readyPromise;
  setInputs(params);
  return parsePythonJson(`
_params = json.loads(_params_json)
_NP = 2 ** int(_params['n_exp'])
_pulse = fc.generate_pulse(
    _params['energy_mJ'], _params['tl_fwhm_fs'], _params['wavelength_nm'],
    _params['GDD_fs2'], _params['TOD_fs3'], _NP, _params['XN'],
    pulse_shape=_params.get('pulse_shape', 'Gaussian'))
(_Ut, _Uf, _pp, _ph, _sp, _sph, _P0, _t, _f, _f0, _delf, _delt) = _pulse

if _params['gas'] == 'Manual':
    _coeff = {
        'beta2_fs2_m': float(_params['beta2_fs2_m']),
        'beta3_fs3_m': float(_params['beta3_fs3_m']),
        'alpha_1_m': float(_params['alpha_1_m']),
        'gamma_1_W_m': float(_params['gamma_1_W_m']),
    }
else:
    _coeff = fc.display_gas_coefficients(
        _params['gas'], _params['wavelength_nm'], _params['pressure'],
        _params['radius_um'])

def _sig(power, threshold, pad=True):
    p = np.asarray(power, dtype=float)
    if p.size == 0 or np.max(p) <= 0:
        return 0, max(0, len(p)-1)
    idx = np.where(p / np.max(p) > threshold)[0]
    if idx.size == 0:
        return 0, len(p)-1
    a, b = int(idx[0]), int(idx[-1])
    if pad:
        w = b-a
        a = max(0, a-w//2)
        b = min(len(p)-1, b+w//2)
    return a, b

def _thin(a, b, maximum=1600):
    n = b-a+1
    if n <= maximum:
        return np.arange(a, b+1, dtype=int)
    return np.unique(np.rint(np.linspace(a, b, maximum)).astype(int))

_t0, _t1 = _sig(_pp, 0.1)
_ti = _thin(_t0, _t1)
_s0, _s1 = _sig(_sp, 0.1)
_si = _thin(_s0, _s1)
_nu = (_f0 - _f) * 1e-12
_order = np.argsort(_nu[_si])
_si = _si[_order]

_payload = {
  'coefficients': {k: float(v) for k, v in _coeff.items()},
  'P0_GW': float(_P0*1e-9),
  'time': {
      'fs': (_t[_ti]*1e15).astype(float).tolist(),
      'power_mJ_fs': (_pp[_ti]*1e-12).astype(float).tolist(),
      'phase_rad': np.angle(_Ut[_ti]).astype(float).tolist(),
  },
  'spectrum': {
      'THz': _nu[_si].astype(float).tolist(),
      'nm': (fc.C_LIGHT / (_nu[_si]*1e12) * 1e9).astype(float).tolist(),
      'power_mW_THz': (_sp[_si]*1e15).astype(float).tolist(),
      'phase_rad': np.angle(_Uf[_si]).astype(float).tolist(),
  },
}
json.dumps(_payload, allow_nan=False)
`);
}

async function runSimulation(params) {
  await readyPromise;
  setInputs(params);
  pyodide.globals.set("_progress_js", progressBridge("run"));

  return parsePythonJson(`
_params = json.loads(_params_json)
def _progress(p):
    _progress_js(int(p))

_last_res = fc.run_simulation(_params, progress_cb=_progress)
_last_run_params = dict(_params)
_r = _last_res

if _params['gas'] == 'Manual':
    _coeff = {
        'beta2_fs2_m': float(_params['beta2_fs2_m']),
        'beta3_fs3_m': float(_params['beta3_fs3_m']),
        'alpha_1_m': float(_params['alpha_1_m']),
        'gamma_1_W_m': float(_params['gamma_1_W_m']),
    }
else:
    _coeff = fc.display_gas_coefficients(
        _params['gas'], _params['wavelength_nm'], _params['pressure'],
        _params['radius_um'])

def _sig(power, threshold, pad=True):
    p = np.asarray(power, dtype=float)
    if p.size == 0 or np.max(p) <= 0:
        return 0, max(0, len(p)-1)
    idx = np.where(p / np.max(p) > threshold)[0]
    if idx.size == 0:
        return 0, len(p)-1
    a, b = int(idx[0]), int(idx[-1])
    if pad:
        w = b-a
        a = max(0, a-w//2)
        b = min(len(p)-1, b+w//2)
    return a, b

def _thin_range(a, b, maximum=1800):
    n = b-a+1
    if n <= maximum:
        return np.arange(a, b+1, dtype=int)
    return np.unique(np.rint(np.linspace(a, b, maximum)).astype(int))

def _finite_vector(a):
    return [float(x) if np.isfinite(x) else None for x in np.asarray(a, dtype=float)]

def _finite_matrix(a):
    aa = np.asarray(a, dtype=float)
    return [[float(x) if np.isfinite(x) else None for x in row] for row in aa]

_t = np.asarray(_r['t'])
_f = np.asarray(_r['f'])
_f0 = float(_r['f0'])
_nu = (_f0 - _f) * 1e-12
_nm = np.full_like(_nu, np.nan, dtype=float)
_pos = _nu > 0
_nm[_pos] = fc.C_LIGHT / (_nu[_pos]*1e12) * 1e9

_rt0, _rt1 = _sig(_r['pulse_power_out'], 0.1)
_rf0, _rf1 = _sig(_r['spec_power_out'], 0.1)
_tl0, _tl1 = _sig(_r['pulse_power_out_TL'], 0.01)

_ti = _thin_range(_rt0, _rt1)
_tli = _thin_range(_tl0, _tl1)
_si = _thin_range(_rf0, _rf1)
_sorder = np.argsort(_nu[_si])
_si = _si[_sorder]

_time_power = np.asarray(_r['pulse_power_out'])[_ti]
_time_mask = _time_power >= 0.01 * float(np.max(_r['pulse_power_out']))
_spec_power = np.asarray(_r['spec_power_out'])[_si]
_spec_mask = _spec_power >= 0.001 * float(np.max(_r['spec_power_out']))

_phase_t = np.asarray(_r['pulse_phase_out'])[_ti]
_phase_s = np.unwrap(np.asarray(_r['spec_phase_out']))[_si]

def _masked(values, mask):
    return [float(v) if bool(m) and np.isfinite(v) else None
            for v, m in zip(np.asarray(values), np.asarray(mask, dtype=bool))]

_raw_display = {
    'suffix': 'raw fiber output',
    'time': {
        'fs': (_t[_ti]*1e15).astype(float).tolist(),
        'power_mJ_fs': (np.asarray(_r['pulse_power_out'])[_ti]*1e-12).astype(float).tolist(),
        'phase_rad': _masked(_phase_t, _time_mask),
        'chirp_THz': _masked(np.asarray(_r['chirp'])[_ti]*1e-12, _time_mask),
    },
    'spectrum': {
        'THz': _nu[_si].astype(float).tolist(),
        'nm': _finite_vector(_nm[_si]),
        'power_mW_THz': (np.asarray(_r['spec_power_out'])[_si]*1e15).astype(float).tolist(),
        'phase_rad': _masked(np.angle(np.asarray(_r['Uf'])[_si]), _spec_mask),
        'group_delay_fs': _masked(np.asarray(_r['gdd_out'])[_si]*1e15, _spec_mask),
    },
}

_tl = {
    'fs': (_t[_tli]*1e15).astype(float).tolist(),
    'power_mJ_fs': (np.asarray(_r['pulse_power_out_TL'])[_tli]*1e-12).astype(float).tolist(),
}

# Phase-analysis display uses the padded significant spectral range.
_phase_display = {
    'THz': _nu[_si].astype(float).tolist(),
    'nm': _finite_vector(_nm[_si]),
    'phase_unwrapped_rad': _finite_vector(np.asarray(_r['phase_unwrapped'])[_si]),
    'power_mW_THz': (np.asarray(_r['spec_power_out'])[_si]*1e15).astype(float).tolist(),
}
_thz_limits = sorted((float(_nu[_rf0]), float(_nu[_rf1])))
_nm_limits = []
if np.isfinite(_nm[_rf0]) and np.isfinite(_nm[_rf1]):
    _nm_limits = sorted((float(_nm[_rf0]), float(_nm[_rf1])))

# Saved propagation maps. Spectral evolution uses the same significant range.
_mi = _thin_range(_rf0, _rf1, maximum=900)
_morder = np.argsort(_nu[_mi])
_mi = _mi[_morder]
_spectral_map = {
    'THz': _nu[_mi].astype(float).tolist(),
    'nm': _finite_vector(_nm[_mi]),
    'z_cm': (np.asarray(_r['spectral_z_m'])*1e2).astype(float).tolist(),
    'power_mW_THz': _finite_matrix(np.asarray(_r['spectral_power_z'])[:, _mi]*1e15),
}

_temp_z = np.asarray(_r['temporal_power_z'])
_env = np.nanmax(_temp_z, axis=0)
_zt0, _zt1 = _sig(_env, 0.01)
_zti = _thin_range(_zt0, _zt1, maximum=900)
_temporal_map = {
    'fs': (_t[_zti]*1e15).astype(float).tolist(),
    'z_cm': (np.asarray(_r['spectral_z_m'])*1e2).astype(float).tolist(),
    'power_mJ_fs': _finite_matrix(_temp_z[:, _zti]*1e-12),
}

_payload = {
    'metrics': {
        'out_energy_mJ': float(_r['out_energy_mJ']),
        'transmission': float(_r['transmission']),
        'broadening': float(_r['broadening']),
        'tl_fwhm_fs': float(_r['tl_fwhm']),
        'P0_GW': float(_r['P0']*1e-9),
        'grid_warning': bool(_r['error_bound_f'] or _r['error_bound_t']),
        'grid_warning_frequency': bool(_r['error_bound_f']),
        'grid_warning_time': bool(_r['error_bound_t']),
        'warning_percent': int(_r['perc_reached']),
    },
    'coefficients': {k: float(v) for k, v in _coeff.items()},
    'raw_display': _raw_display,
    'tl': _tl,
    'phase_display': _phase_display,
    'phase_default_limits': {'THz': _thz_limits, 'nm': _nm_limits},
    'spectral_map': _spectral_map,
    'temporal_map': _temporal_map,
    'n_z_spectra_saved': int(_r.get('n_z_spectra_saved', len(_r['spectral_z_m']))),
    'runtime': {'NP': int(_r['NP'])},
}
json.dumps(_payload, allow_nan=False)
`);
}

async function fixedGddDisplay(gdd) {
  await readyPromise;
  pyodide.globals.set("_gdd_value", Number(gdd));
  return parsePythonJson(`
if _last_res is None:
    raise ValueError('Run a propagation first.')
_c = fc.apply_gdd_to_output(_last_res, float(_gdd_value))
_r = _last_res
_t = np.asarray(_r['t'])
_f = np.asarray(_r['f'])
_f0 = float(_r['f0'])
_nu = (_f0-_f)*1e-12
_nm = np.full_like(_nu, np.nan, dtype=float)
_pos = _nu > 0
_nm[_pos] = fc.C_LIGHT/(_nu[_pos]*1e12)*1e9

def _sig(power, threshold, pad=True):
    p=np.asarray(power,dtype=float); idx=np.where(p/np.max(p)>threshold)[0]
    if idx.size==0: return 0,len(p)-1
    a,b=int(idx[0]),int(idx[-1])
    if pad:
        w=b-a; a=max(0,a-w//2); b=min(len(p)-1,b+w//2)
    return a,b

def _thin(a,b,maximum=1800):
    n=b-a+1
    if n<=maximum: return np.arange(a,b+1,dtype=int)
    return np.unique(np.rint(np.linspace(a,b,maximum)).astype(int))

def _finite(a): return [float(x) if np.isfinite(x) else None for x in np.asarray(a,dtype=float)]
def _masked(v,m): return [float(x) if bool(ok) and np.isfinite(x) else None for x,ok in zip(np.asarray(v),np.asarray(m,dtype=bool))]

_t0,_t1=_sig(_c['pulse_power_comp'],0.01); _ti=_thin(_t0,_t1)
_s0,_s1=_sig(_c['spec_power_comp'],0.1); _si=_thin(_s0,_s1); _si=_si[np.argsort(_nu[_si])]
_tm=np.asarray(_c['pulse_power_comp'])[_ti] >= 0.01*np.max(_c['pulse_power_comp'])
_sm=np.asarray(_c['spec_power_comp'])[_si] >= 0.001*np.max(_c['spec_power_comp'])
_payload={
 'suffix': 'after %.7g fs² GDD' % float(_gdd_value),
 'time': {
   'fs':(_t[_ti]*1e15).astype(float).tolist(),
   'power_mJ_fs':(np.asarray(_c['pulse_power_comp'])[_ti]*1e-12).astype(float).tolist(),
   'phase_rad':_masked(np.asarray(_c['pulse_phase_comp'])[_ti],_tm),
   'chirp_THz':_masked(np.asarray(_c['chirp_comp'])[_ti]*1e-12,_tm),
 },
 'spectrum': {
   'THz':_nu[_si].astype(float).tolist(),
   'nm':_finite(_nm[_si]),
   'power_mW_THz':(np.asarray(_c['spec_power_comp'])[_si]*1e15).astype(float).tolist(),
   'phase_rad':_masked(np.angle(np.asarray(_c['Uf_comp'])[_si]),_sm),
   'group_delay_fs':_masked(np.asarray(_c['gdd_out_comp'])[_si]*1e15,_sm),
 }
}
json.dumps(_payload,allow_nan=False)
`);
}

async function phaseFit(request) {
  await readyPromise;
  pyodide.globals.set("_phase_request_json", JSON.stringify(request));
  return parsePythonJson(`
if _last_res is None:
    raise ValueError('Run a propagation first.')
_q=json.loads(_phase_request_json)
_a=float(_q['min']); _b=float(_q['max']); _order=int(_q['order']); _unit=_q.get('unit','THz')
if _a >= _b: raise ValueError('The minimum fit limit must be smaller than the maximum.')
if _unit=='nm':
    if _a<=0: raise ValueError('Wavelength limits must be positive.')
    _fa=fc.C_LIGHT/(_a*1e-9)*1e-12; _fb=fc.C_LIGHT/(_b*1e-9)*1e-12
    _minf,_maxf=sorted((_fa,_fb))
else:
    _minf,_maxf=_a,_b
_coeff,_xthz,_fit=fc.polynomial_fit(_last_res,_minf,_maxf,_order)
_xnm=fc.C_LIGHT/(np.asarray(_xthz)*1e12)*1e9
_payload={'coefficients':[float(x) for x in _coeff], 'x_THz':np.asarray(_xthz,dtype=float).tolist(), 'x_nm':np.asarray(_xnm,dtype=float).tolist(), 'fitted_rad':np.asarray(_fit,dtype=float).tolist()}
json.dumps(_payload,allow_nan=False)
`);
}

async function compressor(mode, request) {
  await readyPromise;
  pyodide.globals.set("_compressor_request_json", JSON.stringify(request));
  return parsePythonJson(`
if _last_res is None:
    raise ValueError('Run a propagation first.')
_q=json.loads(_compressor_request_json)
if '${mode}' == 'optimize':
    _c=fc.optimize_output_gdd(_last_res,metric=str(_q.get('metric','outer_fwhm95')))
else:
    _c=fc.apply_gdd_to_output(_last_res,float(_q.get('gdd_fs2',0.0)))
_r=_last_res
_t=np.asarray(_r['t']); _f=np.asarray(_r['f']); _f0=float(_r['f0'])
_nu=(_f0-_f)*1e-12
_nm=np.full_like(_nu,np.nan,dtype=float); _pos=_nu>0; _nm[_pos]=fc.C_LIGHT/(_nu[_pos]*1e12)*1e9

def _centered(power):
    p=np.asarray(power,dtype=float); i=int(np.argmax(p)); return (_t-_t[i])*1e15

def _halfwidth(power,tc):
    p=np.asarray(power,dtype=float); idx=np.where(p>0.01*np.max(p))[0]
    if idx.size==0: return 100.0
    return max(abs(float(tc[idx[0]])),abs(float(tc[idx[-1]])))

def _thin_mask(x,maximum=1800):
    if len(x)<=maximum: return np.arange(len(x),dtype=int)
    return np.unique(np.rint(np.linspace(0,len(x)-1,maximum)).astype(int))

def _finite(a): return [float(x) if np.isfinite(x) else None for x in np.asarray(a,dtype=float)]

_pout=np.asarray(_r['pulse_power_out']); _pcomp=np.asarray(_c['pulse_power_comp']); _ptl=np.asarray(_r['pulse_power_out_TL'])
_to=_centered(_pout); _tc=_centered(_pcomp); _ttl=_centered(_ptl)
_hw=max(_halfwidth(_pout,_to),_halfwidth(_pcomp,_tc),_halfwidth(_ptl,_ttl))
_mask=(np.abs(_to)<=_hw) | (np.abs(_tc)<=_hw) | (np.abs(_ttl)<=_hw)
_idx=np.where(_mask)[0]
if _idx.size<2: _idx=np.arange(len(_t))
_if=_thin_mask(_idx); _idx=_idx[_if]

_sp=np.asarray(_c['spec_power_comp']); _sig=np.where(_sp/np.max(_sp)>0.1)[0]
if _sig.size:
    _a,_b=int(_sig[0]),int(_sig[-1]); _w=_b-_a; _a=max(0,_a-_w//2); _b=min(len(_sp)-1,_b+_w//2)
else: _a,_b=0,len(_sp)-1
_si=np.arange(_a,_b+1,dtype=int); _si=_si[_thin_mask(_si)]; _si=_si[np.argsort(_nu[_si])]

def _metrics(d): return {k:float(v) for k,v in d.items()}
_payload={
 'gdd_fs2':float(_c['gdd_fs2']),
 'metrics':_metrics(_c['metrics']), 'output_metrics':_metrics(_c['output_metrics']), 'tl_metrics':_metrics(_c['tl_metrics']),
 'time':{
   'output_fs':_to[_idx].astype(float).tolist(),'output_mJ_fs':(_pout[_idx]*1e-12).astype(float).tolist(),
   'compressed_fs':_tc[_idx].astype(float).tolist(),'compressed_mJ_fs':(_pcomp[_idx]*1e-12).astype(float).tolist(),
   'tl_fs':_ttl[_idx].astype(float).tolist(),'tl_mJ_fs':(_ptl[_idx]*1e-12).astype(float).tolist(),
   'half_width_fs':float(_hw)},
 'spectrum':{'THz':_nu[_si].astype(float).tolist(),'nm':_finite(_nm[_si]),'power_mW_THz':(_sp[_si]*1e15).astype(float).tolist(),'phase_rad':np.angle(np.asarray(_c['Uf_comp'])[_si]).astype(float).tolist()},
 'optimization':{
   'metric':_c.get('optimization_metric'), 'value':float(_c.get('optimization_value',np.nan)) if np.isfinite(_c.get('optimization_value',np.nan)) else None,
   'source':_c.get('optimization_source'), 'fit_seed_gdd_fs2':float(_c.get('fit_seed_gdd_fs2',np.nan)) if np.isfinite(_c.get('fit_seed_gdd_fs2',np.nan)) else None
 }
}
json.dumps(_payload,allow_nan=False)
`);
}

async function runSweep(request) {
  await readyPromise;
  pyodide.globals.set("_sweep_request_json", JSON.stringify(request));
  pyodide.globals.set("_sweep_progress_js", progressBridge("sweep"));
  return parsePythonJson(`
_q=json.loads(_sweep_request_json)
_base=dict(_q['base_params']); _values=np.asarray(_q['values'],dtype=float)

def _progress(p): _sweep_progress_js(int(p))
_last_sweep=fc.run_parameter_sweep(
    _base,_q['parameter_key'],_values,
    phase_fit_config=_q.get('phase_fit_config'),
    optimize_metric=_q.get('optimize_metric','outer_fwhm95'),
    progress_cb=_progress)
_s=_last_sweep

def _finite_vec(a): return [float(x) if np.isfinite(x) else None for x in np.asarray(a,dtype=float)]
def _finite_mat(a):
    aa=np.asarray(a,dtype=float)
    return [[float(x) if np.isfinite(x) else None for x in row] for row in aa]
def _thin_cols(axis,matrix,maximum=900):
    axis=np.asarray(axis,dtype=float); matrix=np.asarray(matrix,dtype=float)
    if len(axis)<=maximum: idx=np.arange(len(axis),dtype=int)
    else: idx=np.unique(np.rint(np.linspace(0,len(axis)-1,maximum)).astype(int))
    return axis[idx],matrix[:,idx]

_sf,_sm=_thin_cols(_s['spectral_frequency_THz'],_s['spectral_power_map_W_per_Hz'])
_tf,_tm=_thin_cols(_s['time_fs'],_s['pulse_power_map_W'])
_types=[str(x) for x in np.asarray(_s['boundary_warning_type'],dtype=object)]
_payload={
 'parameter_key':str(_s['parameter_key']), 'parameter_label':str(_q.get('parameter_label', _s['parameter_key'])), 'values':np.asarray(_s['values'],dtype=float).tolist(),
 'phase_gdd_fs2':_finite_vec(_s['phase_gdd_fs2']), 'optimized_gdd_fs2':_finite_vec(_s['optimized_gdd_fs2']),
 'actual_output_fwhm_fs':_finite_vec(_s['actual_output_fwhm_fs']), 'tl_duration_fs':_finite_vec(_s['tl_duration_fs']),
 'optimized_compressed_duration_fs':_finite_vec(_s['optimized_compressed_duration_fs']), 'swept_compressed_duration_fs':_finite_vec(_s['swept_compressed_duration_fs']),
 'boundary_warning':[bool(x) for x in np.asarray(_s['boundary_warning'])], 'boundary_warning_type':_types,
 'boundary_warning_percent':[int(x) for x in np.asarray(_s['boundary_warning_percent'])],
 'phase_fit_min_THz':_finite_vec(_s['phase_fit_min_THz']), 'phase_fit_max_THz':_finite_vec(_s['phase_fit_max_THz']),
 'spectral_map':{'THz':_sf.astype(float).tolist(),'power_mW_THz':_finite_mat(_sm*1e15)},
 'temporal_map':{'fs':_tf.astype(float).tolist(),'power_GW':_finite_mat(_tm*1e-9)},
 'single_pass_length_sweep':bool(_s.get('single_pass_length_sweep',False)), 'single_pass_output_gdd_sweep':bool(_s.get('single_pass_output_gdd_sweep',False)),
 'optimize_metric':str(_s.get('optimize_metric','outer_fwhm95'))
}
json.dumps(_payload,allow_nan=False)
`);
}

async function sweepCompression(request) {
  await readyPromise;
  pyodide.globals.set("_sweep_compression_json", JSON.stringify(request));
  return parsePythonJson(`
if _last_sweep is None: raise ValueError('Run a parameter sweep first.')
_q=json.loads(_sweep_compression_json)
_c=fc.build_sweep_compressed_temporal_map(_last_sweep,mode=_q.get('mode','optimized'),fixed_gdd_fs2=float(_q.get('fixed_gdd_fs2',0.0)),metric=_q.get('metric'))
_axis=np.asarray(_c['time_fs'],dtype=float); _map=np.asarray(_c['pulse_power_map_W'],dtype=float)
if len(_axis)>900:
    _idx=np.unique(np.rint(np.linspace(0,len(_axis)-1,900)).astype(int)); _axis=_axis[_idx]; _map=_map[:,_idx]
def _fm(a): return [[float(x) if np.isfinite(x) else None for x in row] for row in np.asarray(a,dtype=float)]
def _fv(a): return [float(x) if np.isfinite(x) else None for x in np.asarray(a,dtype=float)]
_payload={'mode':str(_c['mode']),'applied_gdd_fs2':_fv(_c['applied_gdd_fs2']),'duration_fs':_fv(_c['duration_fs']),'time_fs':_axis.astype(float).tolist(),'power_GW':_fm(_map*1e-9)}
json.dumps(_payload,allow_nan=False)
`);
}


async function ensureH5py() {
  if (h5pyLoaded) return;
  send("status", { message: "Loading HDF5 exporter…" });
  await pyodide.loadPackage("h5py");
  h5pyLoaded = true;
}

async function exportH5(scope, request = {}) {
  await readyPromise;
  await ensureH5py();
  pyodide.globals.set("_export_request_json", JSON.stringify(request || {}));
  pyodide.globals.set("_export_scope", String(scope));
  const path = `/tmp/fibdisp_${scope}_export.h5`;
  pyodide.globals.set("_export_path", path);

  pyodide.runPython(`
import h5py
from datetime import datetime, timezone

_q = json.loads(_export_request_json or '{}')
_scope = str(_export_scope)
_path = str(_export_path)
_str_dt = h5py.string_dtype(encoding='utf-8')

def _safe_name(name):
    return str(name).replace('/', '_')

def _write_value(group, name, value):
    name = _safe_name(name)
    if isinstance(value, dict):
        sub = group.create_group(name)
        for k, v in value.items():
            _write_value(sub, k, v)
        return
    if value is None:
        group.attrs[name] = 'null'
        return
    if isinstance(value, str):
        group.create_dataset(name, data=np.asarray(value, dtype=_str_dt))
        return
    if isinstance(value, (bool, np.bool_, int, np.integer, float, np.floating, complex, np.complexfloating)):
        group.attrs[name] = value
        return
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value)
        except Exception:
            group.create_dataset(name, data=np.asarray(json.dumps(value, default=str), dtype=_str_dt))
            return
    else:
        group.create_dataset(name, data=np.asarray(json.dumps(value, default=str), dtype=_str_dt))
        return

    if arr.dtype.kind in ('O', 'U', 'S'):
        flat = arr.ravel()
        try:
            strings = np.asarray([str(x) for x in flat], dtype=_str_dt).reshape(arr.shape)
            group.create_dataset(name, data=strings)
        except Exception:
            group.create_dataset(name, data=np.asarray(json.dumps(value, default=str), dtype=_str_dt))
        return

    kwargs = {}
    if arr.ndim > 0 and arr.size >= 64:
        kwargs = dict(compression='gzip', compression_opts=4, shuffle=True)
    group.create_dataset(name, data=arr, **kwargs)

def _coefficients(params):
    if params.get('gas') == 'Manual':
        return {
            'beta2_fs2_m': float(params['beta2_fs2_m']),
            'beta3_fs3_m': float(params['beta3_fs3_m']),
            'alpha_1_m': float(params['alpha_1_m']),
            'gamma_1_W_m': float(params['gamma_1_W_m']),
        }
    return fc.display_gas_coefficients(
        params['gas'], params['wavelength_nm'], params['pressure'], params['radius_um'])

with h5py.File(_path, 'w') as h5:
    h5.attrs['format'] = 'FibDisp HDF5'
    h5.attrs['format_version'] = '1.0'
    h5.attrs['author'] = 'Davide Faccialà'
    h5.attrs['export_utc'] = datetime.now(timezone.utc).isoformat()
    h5.attrs['export_scope'] = _scope
    h5.attrs['note'] = 'Arrays are full-resolution numerical data; plotting decimation is not used in this file.'

    if _scope == 'single':
        if _last_res is None:
            raise ValueError('Run a propagation before exporting HDF5.')
        _write_value(h5, 'input_parameters', dict(_last_run_params or {}))
        _write_value(h5, 'propagation_coefficients', _coefficients(dict(_last_run_params or {})))
        _write_value(h5, 'result', _last_res)

        _r = _last_res
        _f = np.asarray(_r['f'], dtype=float)
        _f0 = float(_r['f0'])
        _nu_hz = _f0 - _f
        _wl_nm = np.full_like(_nu_hz, np.nan, dtype=float)
        _positive = _nu_hz > 0
        _wl_nm[_positive] = fc.C_LIGHT / _nu_hz[_positive] * 1e9
        _write_value(h5, 'convenience_axes', {
            'time_fs': np.asarray(_r['t'], dtype=float) * 1e15,
            'fft_frequency_Hz': _f,
            'physical_frequency_Hz': _nu_hz,
            'physical_frequency_THz': _nu_hz * 1e-12,
            'wavelength_nm': _wl_nm,
            'propagation_z_cm': np.asarray(_r['spectral_z_m'], dtype=float) * 1e2,
        })

        _post = _q.get('post_display') or {}
        if str(_post.get('mode', 'Raw')) == 'Fixed':
            _post_result = fc.apply_gdd_to_output(_r, float(_post.get('gdd_fs2', 0.0)))
            _write_value(h5, 'selected_fixed_gdd_display', _post_result)

        _phase = _q.get('phase_analysis')
        if _phase:
            _a = float(_phase['min']); _b = float(_phase['max']); _order = int(_phase['order'])
            _unit = str(_phase.get('unit', 'THz'))
            if _unit == 'nm':
                _fa = fc.C_LIGHT / (_a * 1e-9) * 1e-12
                _fb = fc.C_LIGHT / (_b * 1e-9) * 1e-12
                _minf, _maxf = sorted((_fa, _fb))
            else:
                _minf, _maxf = _a, _b
            _coeff, _xthz, _fit = fc.polynomial_fit(_r, _minf, _maxf, _order)
            _write_value(h5, 'phase_analysis', {
                'request': _phase,
                'coefficients': np.asarray(_coeff, dtype=float),
                'frequency_THz': np.asarray(_xthz, dtype=float),
                'fitted_phase_rad': np.asarray(_fit, dtype=float),
            })

        _comp = _q.get('compressor')
        if _comp:
            _gdd = float(_comp.get('gdd_fs2', 0.0))
            _comp_full = fc.apply_gdd_to_output(_r, _gdd)
            _write_value(h5, 'compressor', _comp_full)
            _write_value(h5, 'compressor_web_summary', _comp)

        _write_value(h5, 'web_export_state', _q)

    elif _scope == 'sweep':
        if _last_sweep is None:
            raise ValueError('Run a parameter sweep before exporting HDF5.')
        _write_value(h5, 'settings_snapshot', _q.get('settings_snapshot') or {})
        _write_value(h5, 'sweep', _last_sweep)
        _view = _q.get('compressed_view')
        if _view:
            _compressed = fc.build_sweep_compressed_temporal_map(
                _last_sweep,
                mode=str(_view.get('mode', 'optimized')),
                fixed_gdd_fs2=float(_view.get('fixed_gdd_fs2', 0.0)),
                metric=str(_view.get('metric') or _last_sweep.get('optimize_metric', 'outer_fwhm95')))
            _write_value(h5, 'selected_compressed_view', _compressed)
        _write_value(h5, 'web_export_state', _q)
    else:
        raise ValueError('Unknown HDF5 export scope: %s' % _scope)
`);

  const bytes = pyodide.FS.readFile(path).slice();
  try { pyodide.FS.unlink(path); } catch (_) {}
  self.postMessage({ type: "export-h5-result", scope, bytes }, [bytes.buffer]);
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  try {
    await readyPromise;
    let result;
    if (msg.type === "preview") {
      result = await previewInput(msg.params); send("preview-result", { result });
    } else if (msg.type === "run") {
      send("status", { message: "Running FibDisp on this computer…" });
      const started = performance.now(); result = await runSimulation(msg.params);
      result.runtime.browser_compute_seconds = (performance.now() - started) / 1000;
      send("run-result", { result });
    } else if (msg.type === "fixed-gdd") {
      result = await fixedGddDisplay(msg.gdd_fs2); send("fixed-gdd-result", { result, requestId: msg.requestId });
    } else if (msg.type === "phase-fit") {
      result = await phaseFit(msg.request); send("phase-fit-result", { result });
    } else if (msg.type === "compressor-apply") {
      result = await compressor("apply", msg.request); send("compressor-result", { result, optimized: false });
    } else if (msg.type === "compressor-optimize") {
      send("status", { message: "Optimizing output GDD on this computer…" });
      result = await compressor("optimize", msg.request); send("compressor-result", { result, optimized: true });
    } else if (msg.type === "sweep") {
      send("status", { message: "Running parameter sweep on this computer…" });
      const started = performance.now(); result = await runSweep(msg.request);
      result.browser_compute_seconds = (performance.now() - started) / 1000;
      send("sweep-result", { result });
    } else if (msg.type === "sweep-compression") {
      result = await sweepCompression(msg.request); send("sweep-compression-result", { result, request: msg.request });
    } else if (msg.type === "export-h5") {
      await exportH5(msg.scope, msg.request || {});
    } else if (msg.type === "ping") {
      send("ready", { pyodideVersion: pyodide.version });
    }
  } catch (error) {
    console.error(error);
    const full = error?.stack || error?.message || String(error);
    const message = error?.message || String(error);
    send("error", { scope: msg.type || "worker", message, details: full });
  }
};
