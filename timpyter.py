import streamlit as st
import io
import contextlib
import ast
import json

st.set_page_config("Streamlit Notebook")
st.title("📝 Streamlit Python Notebook")


# -------- INITIAL SESSION STATE --------
if "kernel" not in st.session_state:
    st.session_state.kernel = {}

if "cells" not in st.session_state:
    st.session_state.cells = ["print('Hello Notebook!')"]


# --------- SAVE / LOAD NOTEBOOK ----------
def save_notebook():
    data = {
        "cells": st.session_state.cells
    }
    json_data = json.dumps(data, indent=2)
    st.download_button(
        "💾 Download Notebook",
        json_data,
        file_name="notebook.json",
        mime="application/json"
    )


def load_notebook(file):
    content = json.load(file)
    st.session_state.cells = content["cells"]


# ----------- EXECUTION ENGINE -----------
def run_cell(code, idx):

    # Guard stale cell index
    if idx >= len(st.session_state.cells):
        return "⚠️ Cell no longer exists."

    buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(buffer):

            # Run full Python code
            exec(code, st.session_state.kernel)

            # Evaluate final expression (Jupyter style)
            try:
                tree = ast.parse(code)
                last = tree.body[-1]

                if isinstance(last, ast.Expr):
                    compiled = compile(
                        ast.Expression(last.value),
                        "<ast>",
                        "eval"
                    )
                    val = eval(compiled, st.session_state.kernel)
                    if val is not None:
                        print(val)

            except Exception:
                pass

        # Update stored cell code
        st.session_state.cells[idx] = code

        output = buffer.getvalue()
        return output or "(no output)"

    except Exception as e:
        return f"⚠️ Error: {e}"


# ----------- NOTEBOOK TOOLBAR -----------
colA, colB = st.columns(2)

with colA:
    save_notebook()

with colB:
    file = st.file_uploader("📂 Load Notebook", type="json")
    if file:
        load_notebook(file)


st.markdown("---")


# ----------- VARIABLE INSPECTOR -----------
with st.expander("🧭 Variable Inspector", expanded=False):

    vars_table = {
        k: str(v)
        for k, v in st.session_state.kernel.items()
        if not k.startswith("__")
    }

    if vars_table:
        for k, v in vars_table.items():
            st.write(f"**{k}**  →  `{v}`")
    else:
        st.caption("No variables yet.")


st.markdown("---")


# ----------- RENDER CELLS -----------
for i, cell in enumerate(list(st.session_state.cells)):

    st.markdown(f"### 🟦 Cell {i+1}")

    code = st.text_area(
        f"cell_{i}",
        value=cell,
        height=150,
        label_visibility="collapsed"
    )

    c1, c2, c3, c4 = st.columns([1,1,1,1])

    with c1:
        if st.button("▶ Run", key=f"run_{i}"):
            out = run_cell(code, i)
            st.session_state[f"out_{i}"] = out

    with c2:
        if st.button("⬆ Move Up", key=f"up_{i}") and i > 0:
            st.session_state.cells[i-1], st.session_state.cells[i] = \
                st.session_state.cells[i], st.session_state.cells[i-1]

    with c3:
        if st.button("⬇ Move Down", key=f"down_{i}") and i < len(st.session_state.cells)-1:
            st.session_state.cells[i+1], st.session_state.cells[i] = \
                st.session_state.cells[i], st.session_state.cells[i+1]

    with c4:
        if st.button("🗑 Delete", key=f"del_{i}"):
            st.session_state.cells.pop(i)
            st.experimental_rerun()

    st.code(st.session_state.get(f"out_{i}", ""), language="text")
    st.divider()


# ----------- ADD CELL -----------
if st.button("➕ Add Cell"):
    st.session_state.cells.append("")
