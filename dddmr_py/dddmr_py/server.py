"""目标点发布服务 (无 ROS 版的 /move_base_simple/goal).

启动::

    python -m dddmr_py serve map.pcd --start 0 0 0 --port 8000

然后:
  * 浏览器打开 http://localhost:8000 , 在 3D 点云上单击即可发布目标点并看到路径;
  * 或用命令行发布:  python -m dddmr_py goal 12 3 3.0
  * 或用任意 HTTP 客户端 POST /goal {"x":12,"y":3,"z":3.0}
"""

from __future__ import annotations

import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import numpy as np

from .planner import GlobalPlanner3D, Path
from .viz import render_html_string


def _path_payload(path: Path) -> dict:
    return {
        "success": bool(path.success),
        "message": path.message,
        "length": path.length,
        "climb": path.climb,
        "cost": None if not np.isfinite(path.cost) else float(path.cost),
        "expanded": int(path.expanded),
        "planning_time": float(path.planning_time),
        "path": {
            "x": np.round(path.points[:, 0], 4).tolist(),
            "y": np.round(path.points[:, 1], 4).tolist(),
            "z": np.round(path.points[:, 2], 4).tolist(),
            "yaw": np.round(path.yaw, 4).tolist(),
        } if len(path.points) else {"x": [], "y": [], "z": [], "yaw": []},
    }


class PlanningService:
    """把规划器包装成"当前位姿 + 目标点 -> 路径"的服务."""

    def __init__(self, planner: GlobalPlanner3D, robot_pose=(0.0, 0.0, 0.0),
                 map_name: str = "map.pcd"):
        self.planner = planner
        self.robot_pose = np.asarray(robot_pose, dtype=np.float64)
        self.map_name = map_name
        self.last_path: Optional[Path] = None

    def set_pose(self, pose) -> None:
        self.robot_pose = np.asarray(pose, dtype=np.float64)

    def send_goal(self, goal, start=None) -> dict:
        start = self.robot_pose if start is None else np.asarray(start, dtype=np.float64)
        path = self.planner.make_plan(start, goal)
        self.last_path = path
        if path.success:
            # 到达后把机器人位姿更新为终点, 便于连续发布目标点
            self.robot_pose = path.points[-1]
        return _path_payload(path)


def _make_handler(service: PlanningService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "dddmr-py/1.0"

        def log_message(self, fmt, *args):  # 降低日志噪音
            print(f"[server] {self.address_string()} {fmt % args}")

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if not n:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8") or "{}")

        def do_OPTIONS(self):  # noqa: N802
            self._send(204, b"", "text/plain")

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                html = render_html_string(
                    service.planner.ground, service.last_path,
                    start=service.robot_pose, goal=None,
                    map_name=service.map_name, server_mode=True)
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/stats":
                self._json(service.planner.stats())
            elif self.path == "/pose":
                self._json({"pose": service.robot_pose.tolist()})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):  # noqa: N802
            try:
                data = self._read_json()
                if self.path == "/goal":
                    goal = data.get("goal") or [data.get("x"), data.get("y"), data.get("z")]
                    self._json(service.send_goal(goal, data.get("start")))
                elif self.path == "/plan":
                    self._json(service.send_goal(data["goal"], data.get("start")))
                elif self.path == "/pose":
                    service.set_pose(data.get("pose") or [data["x"], data["y"], data["z"]])
                    self._json({"ok": True, "pose": service.robot_pose.tolist()})
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as exc:  # noqa: BLE001
                self._json({"success": False, "message": f"{type(exc).__name__}: {exc}"}, 400)

    return Handler


def serve(planner: GlobalPlanner3D, host: str = "0.0.0.0", port: int = 8000,
          robot_pose=(0.0, 0.0, 0.0), map_name: str = "map.pcd") -> None:
    service = PlanningService(planner, robot_pose, map_name)
    httpd = ThreadingHTTPServer((host, port), _make_handler(service))
    print(f"[server] 规划服务已启动: http://{host}:{port}  (Ctrl+C 停止)")
    print(f"[server] 发布目标点: POST /goal  {{'x':..,'y':..,'z':..}}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] 已停止")
    finally:
        httpd.server_close()


# --------------------------------------------------------------------------
def publish_goal(x: float, y: float, z: float, host: str = "127.0.0.1", port: int = 8000,
                 start=None, timeout: float = 30.0) -> dict:
    """客户端: 向规划服务发布一个 3D 目标点, 返回路径 JSON."""
    payload = {"x": float(x), "y": float(y), "z": float(z)}
    if start is not None:
        payload["start"] = [float(v) for v in start]
    req = urllib.request.Request(
        f"http://{host}:{port}/goal", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())
