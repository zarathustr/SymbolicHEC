#!/usr/bin/env python3
from __future__ import annotations
import pathlib, re, struct, sys
from typing import List, Tuple
ROOT = pathlib.Path(__file__).resolve().parents[1]

def ints_from_array(text: str, name: str) -> List[int]:
    m = re.search(r"\b"+re.escape(name)+r"\s*=\s*\[(.*?)\];", text, re.S)
    if not m: raise RuntimeError(f"missing {name}")
    return [int(x) for x in re.findall(r"-?\d+", m.group(1).replace('...', ' '))]

def parse_solver(path: pathlib.Path, blocks_csv: pathlib.Path, out: pathlib.Path):
    text = path.read_text()
    action = int(re.search(r"complete_x(\d+)", path.name).group(1))
    n = int(re.search(r"C0\s*=\s*spalloc\((\d+),(\d+),", text).group(1))
    m = int(re.search(r"C1\s*=\s*spalloc\((\d+),(\d+),", text).group(2))
    tail = int(re.search(r"RR\s*=\s*\[-C1\(end-(\d+):end", text).group(1)) + 1
    pairs = [(int(a), int(b)) for a, b in re.findall(r"coeffs\((\d+)\)\s*=\s*data\((\d+)\);", text)]
    coeff_map = [0] * max(i for i, _ in pairs)
    for i, j in pairs: coeff_map[i-1] = j-1
    c0_lin = [x-1 for x in ints_from_array(text, 'C0_ind')]
    c1_lin = [x-1 for x in ints_from_array(text, 'C1_ind')]
    c0_coeff = [x-1 for x in ints_from_array(text, 'coeffs0_ind')]
    c1_coeff = [x-1 for x in ints_from_array(text, 'coeffs1_ind')]
    am_ind = [x-1 for x in ints_from_array(text, 'AM_ind')]
    sol_sources=[]
    for k in range(1,7):
        mm = re.search(rf"sols\({k},:\)\s*=\s*(diag\(D\)|V\((\d+),:\))", text)
        sol_sources.append(-1 if mm.group(1)=='diag(D)' else int(mm.group(2))-1)
    blocks=[]
    for line in blocks_csv.read_text().strip().splitlines()[1:]:
        if not line.strip(): continue
        parts=line.split(','); blocks.append((int(parts[1])-1, int(parts[2])))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as f:
        f.write(b'AXYBTPL1')
        f.write(struct.pack('<10I', 1, action, n, m, len(c0_lin), len(c1_lin), len(coeff_map), len(am_ind), tail, len(blocks)))
        f.write(struct.pack(f'<{len(coeff_map)}I', *coeff_map))
        f.write(struct.pack(f'<{len(c0_lin)}Q', *c0_lin))
        f.write(struct.pack(f'<{len(c0_coeff)}I', *c0_coeff))
        f.write(struct.pack(f'<{len(c1_lin)}Q', *c1_lin))
        f.write(struct.pack(f'<{len(c1_coeff)}I', *c1_coeff))
        f.write(struct.pack(f'<{len(am_ind)}I', *am_ind))
        f.write(struct.pack('<6i', *sol_sources))
        flat=[x for b in blocks for x in b]
        f.write(struct.pack(f'<{len(flat)}I', *flat))

def split_commas(s: str) -> List[str]:
    out=[]; start=0; depth=0
    for i,ch in enumerate(s):
        if ch in '([{': depth += 1
        elif ch in ')]}': depth -= 1
        elif ch==',' and depth==0:
            out.append(s[start:i].strip()); start=i+1
    last=s[start:].strip()
    if last: out.append(last)
    return out

def expr_cpp(expr: str) -> str:
    expr=expr.strip()
    expr=re.sub(r"\bin(\d+)\((\d+)\s*,\s*:\)", lambda m: f"in{m.group(1)}[{int(m.group(2))-1}]", expr)
    expr=re.sub(r"\bin(\d+)\((\d+)\)", lambda m: f"in{m.group(1)}[{int(m.group(2))-1}]", expr)
    expr=expr.replace('.*','*').replace('./','/').replace('.+','+').replace('.-','-')
    expr=re.sub(r"\b([A-Za-z_]\w*)\.\^2\b", r"(\1*\1)", expr)
    expr=re.sub(r"\b([A-Za-z_]\w*)\.\^([0-9]+(?:\.[0-9]+)?)", r"std::pow(\1, \2)", expr)
    return expr

def norm_body(path: pathlib.Path) -> List[str]:
    txt=re.sub(r'%.*','',path.read_text()).replace('...','')
    raw=[ln.strip() for ln in txt.splitlines() if ln.strip()]
    if any('ft_1({' in ln for ln in raw):
        out=[]; in_helper=False; skip_unpack=False
        for ln in raw:
            if ln.startswith('function'):
                if 'ft_1' in ln: in_helper=True; skip_unpack=True
                continue
            if 'ft_1({' in ln: continue
            if ln=='end': continue
            if in_helper and skip_unpack and '= ct{:}' in ln:
                skip_unpack=False; continue
            out.append(ln)
        return out
    return [ln for ln in raw if not ln.startswith('function') and ln!='end']

def emit_func(path: pathlib.Path, name: str, arity: int) -> str:
    args=', '.join(f'const std::vector<double>& in{i}' for i in range(1,arity+1))
    cpp=[f'std::vector<double> {name}({args}) {{']; declared=set()
    for ln in norm_body(path):
        if not ln.endswith(';'): raise RuntimeError(f'bad line {path}: {ln[:80]}')
        lhs,rhs=[x.strip() for x in ln[:-1].split('=',1)]
        if rhs.startswith('reshape('):
            m=re.match(r"reshape\(\[(.*)\]\s*,\s*\[?(\d+)\s*,\s*(\d+)\]?\)", rhs)
            items=[]
            for part in split_commas(m.group(1)):
                part=part.strip(); items.append(('vec',part) if re.fullmatch(r'mt\d+',part) else ('expr',expr_cpp(part)))
            cpp += ['    std::vector<double> out;', f'    out.reserve({int(m.group(2))*int(m.group(3))});']
            for kind,item in items:
                cpp.append(f'    out.insert(out.end(), {item}.begin(), {item}.end());' if kind=='vec' else f'    out.push_back({item});')
            cpp.append('    return out;')
        elif rhs.startswith('[') and rhs.endswith(']'):
            parts=split_commas(rhs[1:-1].replace(';',',')); items=[]
            for part in parts:
                part=part.strip(); items.append(('vec',part) if re.fullmatch(r'mt\d+',part) else ('expr',expr_cpp(part)))
            if lhs.startswith('mt'):
                cpp += [f'    std::vector<double> {lhs};', f'    {lhs}.reserve({len(items)});']
                for kind,item in items:
                    cpp.append(f'    {lhs}.insert({lhs}.end(), {item}.begin(), {item}.end());' if kind=='vec' else f'    {lhs}.push_back({item});')
            else:
                cpp += ['    std::vector<double> out;', f'    out.reserve({len(items)});']
                for kind,item in items:
                    cpp.append(f'    out.insert(out.end(), {item}.begin(), {item}.end());' if kind=='vec' else f'    out.push_back({item});')
                cpp.append('    return out;')
        else:
            rhs_cpp=expr_cpp(rhs)
            if lhs in declared: cpp.append(f'    {lhs} = {rhs_cpp};')
            else:
                declared.add(lhs); cpp.append(f'    double {lhs} = {rhs_cpp};')
    cpp.append('}')
    return '\n'.join(cpp)

def gen_symbolics(src: pathlib.Path, out_cpp: pathlib.Path, out_hpp: pathlib.Path):
    specs=[('A_eq_tXtY_func.m','A_eq_tXtY_func',1),('A_res_eq_tXtY_RY_func.m','A_res_eq_tXtY_RY_func',3),('b_eq_tXtY_func.m','b_eq_tXtY_func',2),('W_J_trans_cayley_gY_func.m','W_J_trans_cayley_gY_func',6),('W_J_rot_gXgY_func.m','W_J_rot_gXgY_func',2),('W_AXYB_gXgY_func.m','W_AXYB_gXgY_func',2)]
    out_hpp.write_text('''#pragma once\n#include <vector>\nnamespace axyb {\nstd::vector<double> A_eq_tXtY_func(const std::vector<double>& in1);\nstd::vector<double> A_res_eq_tXtY_RY_func(const std::vector<double>& in1, const std::vector<double>& in2, const std::vector<double>& in3);\nstd::vector<double> b_eq_tXtY_func(const std::vector<double>& in1, const std::vector<double>& in2);\nstd::vector<double> W_J_trans_cayley_gY_func(const std::vector<double>& in1, const std::vector<double>& in2, const std::vector<double>& in3, const std::vector<double>& in4, const std::vector<double>& in5, const std::vector<double>& in6);\nstd::vector<double> W_J_rot_gXgY_func(const std::vector<double>& in1, const std::vector<double>& in2);\nstd::vector<double> W_AXYB_gXgY_func(const std::vector<double>& in1, const std::vector<double>& in2);\n}\n''')
    body=['#include "axyb/generated_symbolics.hpp"','#include <cmath>','#include <vector>','','namespace axyb {','']
    for fname,name,arity in specs:
        body.append(emit_func(src/fname,name,arity)); body.append('')
    body.append('}')
    out_cpp.write_text('\n'.join(body))

def gen_templates(src: pathlib.Path, out: pathlib.Path):
    for a in range(1,7): parse_solver(src/f'solver_AXYB_cayley_complete_x{a}.m', src/'docs'/f'departition_blocks_x{a}.csv', out/f'AXYB_complete_x{a}.tpl')

def main():
    src=pathlib.Path(sys.argv[1]).resolve()
    gen_templates(src, ROOT/'data/templates')
    gen_symbolics(src, ROOT/'src/generated_symbolics.cpp', ROOT/'include/axyb/generated_symbolics.hpp')
if __name__=='__main__': main()
