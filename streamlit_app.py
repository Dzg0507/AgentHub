import time
import io
import os
from typing import Dict, Any, Optional

import requests
import streamlit as st


# Configure API base using env var fallback, persisted in session_state
DEFAULT_API_BASE = os.environ.get("AGENT_HUB_API_BASE", "http://127.0.0.1:8000")
if "api_base" not in st.session_state:
    st.session_state["api_base"] = DEFAULT_API_BASE


def normalize_api_base(base: Optional[str]) -> str:
    b = (base or DEFAULT_API_BASE).strip()
    if not b:
        b = DEFAULT_API_BASE
    if "://" not in b:
        b = f"http://{b}"
    return b.rstrip("/")


def api_url(path: str) -> str:
    base = normalize_api_base(st.session_state.get("api_base"))
    return f"{base}{path}"


def post_json(path: str, payload: Optional[Dict[str, Any]] = None) -> requests.Response:
    url = api_url(path)
    # Allow slower server responses on POST (create project)
    return requests.post(url, json=payload or {}, timeout=(5, 300))


def get_json(path: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
    url = api_url(path)
    return requests.get(url, params=params or {}, timeout=60)


def delete(path: str) -> requests.Response:
    url = api_url(path)
    return requests.delete(url, timeout=60)


def download_zip(project_id: str) -> Optional[bytes]:
    url = api_url(f"/projects/{project_id}/download")
    try:
        r = requests.get(url, timeout=300)
        if r.status_code == 200:
            return r.content
    except requests.exceptions.RequestException as e:
        st.warning(f"Download failed: {e}")
    return None


def render_project_row(project: Dict[str, Any]):
    project_id = project.get("project_id")
    status = project.get("status")
    created_at = project.get("created_at")
    updated_at = project.get("updated_at")
    files_created = project.get("files_created", [])

    cols = st.columns([2, 2, 2, 1.2, 2.2, 2.2])
    with cols[0]:
        st.markdown(f"**{project_id}**")
        st.caption(f"Created: {created_at}\nUpdated: {updated_at}")
    with cols[1]:
        st.markdown(f"Status: **{status}**")
        if status in ("PENDING", "PROCESSING"):
            st.progress(0.5)
    with cols[2]:
        st.markdown(f"Files: {len(files_created)}")
        if files_created:
            with st.expander("View files"):
                for f in files_created:
                    st.code(f)
    with cols[3]:
        if st.button("Logs", key=f"logs-{project_id}"):
            r = get_json(f"/projects/{project_id}/logs")
            if r.ok:
                raw_logs = r.json().get("execution_log", [])
                # Normalize to a list of strings
                if isinstance(raw_logs, bool):
                    norm_logs = []
                elif isinstance(raw_logs, (list, tuple)):
                    norm_logs = [str(x) for x in raw_logs]
                else:
                    norm_logs = [str(raw_logs)] if raw_logs is not None else []
                st.session_state[f"logs-{project_id}"] = norm_logs
        logs = st.session_state.get(f"logs-{project_id}")
        if logs is not None:
            with st.expander("Execution Log", expanded=False):
                # Guard: ensure logs are iterable
                if isinstance(logs, bool):
                    logs_iter = []
                elif isinstance(logs, (list, tuple)):
                    logs_iter = logs
                else:
                    logs_iter = [str(logs)]
                for line in logs_iter:
                    st.text(line)
    with cols[4]:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Retry", key=f"retry-{project_id}"):
                r = post_json(f"/projects/{project_id}/retry")
                if not r.ok:
                    st.error(f"Retry failed: {r.status_code} {r.text}")
        with c2:
            if st.button("Download", key=f"download-{project_id}"):
                content = download_zip(project_id)
                if content:
                    st.download_button(
                        label="Save ZIP",
                        data=content,
                        file_name=f"{project_id}.zip",
                        mime="application/zip",
                        key=f"dlbtn-{project_id}",
                    )
                else:
                    st.warning("No zip available yet.")
    with cols[5]:
        d1, d2 = st.columns(2)
        with d1:
            if st.button("Force Scaffold", key=f"scaf-{project_id}"):
                r = post_json(f"/projects/{project_id}/force-scaffold")
                if r.ok:
                    st.success("Scaffold created")
                else:
                    st.error(f"Scaffold failed: {r.status_code} {r.text}")
        with d2:
            if st.button("Delete", type="secondary", key=f"delete-{project_id}"):
                r = delete(f"/projects/{project_id}")
                if not r.ok:
                    st.error(f"Delete failed: {r.status_code} {r.text}")


def main():
    st.set_page_config(page_title="Agent Control Hub", layout="wide")
    st.title("Agent Control Hub UI")
    st.caption("Streamlit front-end for your FastAPI multi-agent backend")

    with st.sidebar:
        st.subheader("Server")
        st.text_input("API Base", key="api_base_input", value=st.session_state.get("api_base", DEFAULT_API_BASE))
        if st.button("Use server"):
            # Persist input to the effective base and reload
            st.session_state["api_base"] = st.session_state.get("api_base_input", DEFAULT_API_BASE)
            st.rerun()

        st.divider()
        st.subheader("Create Project")
        prompt = st.text_area("Prompt", height=150, placeholder="Describe what to build...")
        if st.button("Create Project", use_container_width=True, type="primary"):
            if not prompt.strip():
                st.warning("Prompt cannot be empty")
            else:
                with st.status("Creating project...", expanded=False) as status:
                    try:
                        r = post_json("/projects", {"prompt": prompt.strip()})
                        if r.ok:
                            data = r.json()
                            status.update(label=f"Created project {data.get('project_id')}", state="complete")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            status.update(label=f"Create failed: {r.status_code}", state="error")
                            st.error(f"Create failed: {r.status_code} {r.text}")
                    except requests.exceptions.RequestException as e:
                        status.update(label="Request timed out or failed", state="error")
                        st.error(f"Request failed: {e}")

    tabs = st.tabs(["Projects", "Health", "Debug"])

    with tabs[0]:
        auto_refresh = st.toggle("Auto refresh", value=True)
        refresh_seconds = st.slider("Refresh interval (sec)", 1, 15, 3)
        resp = get_json("/projects")
        if not resp.ok:
            st.error(f"Failed to list projects: {resp.status_code} {resp.text}")
        else:
            payload = resp.json()
            projects = payload.get("projects", [])
            st.write(f"Total: {len(projects)}")
            for project in projects:
                with st.container(border=True):
                    render_project_row(project)
            if auto_refresh:
                time.sleep(refresh_seconds)
                st.rerun()

    with tabs[1]:
        r = get_json("/health")
        if r.ok:
            health = r.json()
            st.json(health)
        else:
            st.error(f"Health check failed: {r.status_code} {r.text}")

    with tabs[2]:
        st.write("Two-agent quick test")
        dbg_prompt = st.text_input("Prompt", key="dbg_prompt")
        if st.button("Run Debug Two-Agent"):
            r = post_json("/debug/two-agent", {"prompt": dbg_prompt})
            if r.ok:
                st.json(r.json())
            else:
                st.error(f"Debug failed: {r.status_code} {r.text}")


if __name__ == "__main__":
    main()


