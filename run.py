import streamlit as st
import streamlit.web.cli
import streamlit.runtime.scriptrunner
import subprocess
import os

def run_streamlit():
    script_path = os.path.join(os.path.dirname(__file__), "main2.py")
    subprocess.run(["streamlit", "run", script_path])
    #subprocess.run(["python", script_path])


if __name__ == "__main__":
    run_streamlit()


