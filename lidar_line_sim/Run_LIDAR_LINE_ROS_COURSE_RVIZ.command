#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SSH_CONFIG="${AUTONAV_LIMA_SSH_CONFIG:-$HOME/.lima/autonav-ros22/ssh.config}"
SSH_HOST="${AUTONAV_LIMA_SSH_HOST:-lima-autonav-ros22}"
REMOTE_SIM_REPO_REL="${AUTONAV_REMOTE_SIM_REPO_REL:-autonav-work/AutoNav-Simulated-Universe}"
REMOTE_AUTONAV_REPO_REL="${AUTONAV_REMOTE_REPO_REL:-autonav-work/AutoNav_25-26}"
LOCAL_VNC_PORT="${AUTONAV_RVIZ_LOCAL_VNC_PORT:-5902}"
REMOTE_VNC_PORT="${AUTONAV_RVIZ_REMOTE_VNC_PORT:-5902}"
VNC_PASSWORD="${AUTONAV_RVIZ_VNC_PASSWORD:-autonav}"
RVIZ_DISPLAY="${AUTONAV_RVIZ_DISPLAY:-:99}"
RVIZ_GEOMETRY="${AUTONAV_RVIZ_GEOMETRY:-1280x800x24}"
RVIZ_CONFIG_LOCAL="$SCRIPT_DIR/rviz/lidar_line_course.rviz"

if [[ ! -f "$SSH_CONFIG" ]]; then
  echo "Lima SSH config not found: $SSH_CONFIG" >&2
  echo "Start the autonav-ros22 VM first." >&2
  exit 1
fi

if [[ ! -f "$RVIZ_CONFIG_LOCAL" ]]; then
  echo "RViz config not found: $RVIZ_CONFIG_LOCAL" >&2
  exit 1
fi

remote_cleanup() {
  ssh -S none -o ControlMaster=no -F "$SSH_CONFIG" "$SSH_HOST" \
    'SESSION_DIR="${XDG_RUNTIME_DIR:-/tmp}/autonav-rviz-vnc";
     for f in "$SESSION_DIR"/rviz.pid "$SESSION_DIR"/x11vnc.pid "$SESSION_DIR"/openbox.pid "$SESSION_DIR"/xvfb.pid; do
       if [ -f "$f" ]; then
         pid="$(cat "$f" 2>/dev/null || true)";
         if [ -n "$pid" ]; then kill "$pid" 2>/dev/null || true; fi;
         rm -f "$f";
       fi;
     done' >/dev/null 2>&1 || true
}

ssh_pid=""
cleanup_ran=0
local_cleanup() {
  if [[ "$cleanup_ran" -eq 1 ]]; then
    return
  fi
  cleanup_ran=1
  remote_cleanup
  if [[ -n "${ssh_pid:-}" ]] && kill -0 "$ssh_pid" 2>/dev/null; then
    kill "$ssh_pid" 2>/dev/null || true
    wait "$ssh_pid" 2>/dev/null || true
  fi
}

handle_signal() {
  local_cleanup
  exit 130
}

trap local_cleanup EXIT
trap handle_signal INT TERM

echo "Syncing RViz config to $SSH_HOST:$REMOTE_SIM_REPO_REL ..."
ssh -S none -o ControlMaster=no -F "$SSH_CONFIG" "$SSH_HOST" \
  "mkdir -p \"\$HOME/$REMOTE_SIM_REPO_REL/lidar_line_sim/rviz\""
scp -F "$SSH_CONFIG" "$RVIZ_CONFIG_LOCAL" \
  "$SSH_HOST:$REMOTE_SIM_REPO_REL/lidar_line_sim/rviz/lidar_line_course.rviz"

echo
echo "Starting VM RViz VNC session."
echo "Local VNC URL: vnc://localhost:$LOCAL_VNC_PORT"
echo "VNC password: $VNC_PASSWORD"
echo "Keep this terminal open; Ctrl-C stops RViz and the VNC tunnel."
echo

if [[ "${AUTONAV_OPEN_VNC:-1}" != "0" ]] && command -v open >/dev/null 2>&1; then
  (
    sleep 4
    open "vnc://localhost:$LOCAL_VNC_PORT" >/dev/null 2>&1 || true
  ) &
fi

ssh -T \
  -S none \
  -o ControlMaster=no \
  -L "$LOCAL_VNC_PORT:127.0.0.1:$REMOTE_VNC_PORT" \
  -F "$SSH_CONFIG" \
  "$SSH_HOST" \
  "REMOTE_SIM_REPO_REL='$REMOTE_SIM_REPO_REL' \
   REMOTE_AUTONAV_REPO_REL='$REMOTE_AUTONAV_REPO_REL' \
   REMOTE_VNC_PORT='$REMOTE_VNC_PORT' \
   VNC_PASSWORD='$VNC_PASSWORD' \
   RVIZ_DISPLAY='$RVIZ_DISPLAY' \
   RVIZ_GEOMETRY='$RVIZ_GEOMETRY' \
   bash -s" <<'REMOTE' &
set -euo pipefail

SESSION_DIR="${XDG_RUNTIME_DIR:-/tmp}/autonav-rviz-vnc"
mkdir -p "$SESSION_DIR"

REMOTE_SIM_REPO="$HOME/${REMOTE_SIM_REPO_REL:-autonav-work/AutoNav-Simulated-Universe}"
REMOTE_AUTONAV_REPO="$HOME/${REMOTE_AUTONAV_REPO_REL:-autonav-work/AutoNav_25-26}"
RVIZ_CONFIG="$REMOTE_SIM_REPO/lidar_line_sim/rviz/lidar_line_course.rviz"
DISPLAY_ID="${RVIZ_DISPLAY#:}"

stop_pid_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(cat "$file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      for _ in {1..20}; do
        if ! kill -0 "$pid" 2>/dev/null; then
          break
        fi
        sleep 0.1
      done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$file"
  fi
}

cleanup_previous() {
  stop_pid_file "$SESSION_DIR/rviz.pid"
  stop_pid_file "$SESSION_DIR/x11vnc.pid"
  stop_pid_file "$SESSION_DIR/openbox.pid"
  stop_pid_file "$SESSION_DIR/xvfb.pid"
  if [[ -e "/tmp/.X${DISPLAY_ID}-lock" ]] && ! pgrep -f "Xvfb ${RVIZ_DISPLAY}" >/dev/null 2>&1; then
    rm -f "/tmp/.X${DISPLAY_ID}-lock"
  fi
}

cleanup() {
  stop_pid_file "$SESSION_DIR/rviz.pid"
  stop_pid_file "$SESSION_DIR/x11vnc.pid"
  stop_pid_file "$SESSION_DIR/openbox.pid"
  stop_pid_file "$SESSION_DIR/xvfb.pid"
}

cleanup_previous
trap cleanup EXIT INT TERM

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS Humble setup not found at /opt/ros/humble/setup.bash" >&2
  exit 1
fi

set +u
source /opt/ros/humble/setup.bash
if [[ -f "$REMOTE_AUTONAV_REPO/isaac_ros-dev/install/setup.bash" ]]; then
  source "$REMOTE_AUTONAV_REPO/isaac_ros-dev/install/setup.bash"
fi
set -u

if ! command -v Xvfb >/dev/null 2>&1 || ! command -v x11vnc >/dev/null 2>&1; then
  echo "Missing Xvfb/x11vnc. Install xvfb x11vnc openbox in the VM." >&2
  exit 1
fi

if ! command -v rviz2 >/dev/null 2>&1; then
  echo "rviz2 is not installed in the VM. Install ros-humble-rviz2." >&2
  exit 1
fi

if [[ ! -f "$RVIZ_CONFIG" ]]; then
  echo "RViz config not found in VM: $RVIZ_CONFIG" >&2
  exit 1
fi

Xvfb "$RVIZ_DISPLAY" -screen 0 "$RVIZ_GEOMETRY" +extension GLX +render -noreset \
  >"$SESSION_DIR/xvfb.log" 2>&1 &
echo $! > "$SESSION_DIR/xvfb.pid"

for _ in {1..50}; do
  if DISPLAY="$RVIZ_DISPLAY" xdpyinfo >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

DISPLAY="$RVIZ_DISPLAY" openbox >"$SESSION_DIR/openbox.log" 2>&1 &
echo $! > "$SESSION_DIR/openbox.pid"

x11vnc \
  -display "$RVIZ_DISPLAY" \
  -localhost \
  -rfbport "${REMOTE_VNC_PORT:-5901}" \
  -forever \
  -shared \
  -passwd "${VNC_PASSWORD:-autonav}" \
  -noxdamage \
  -repeat \
  >"$SESSION_DIR/x11vnc.log" 2>&1 &
echo $! > "$SESSION_DIR/x11vnc.pid"

export DISPLAY="$RVIZ_DISPLAY"
export LIBGL_ALWAYS_SOFTWARE=1
export QT_X11_NO_MITSHM=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

rviz2 -d "$RVIZ_CONFIG" >"$SESSION_DIR/rviz.log" 2>&1 &
rviz_pid=$!
echo "$rviz_pid" > "$SESSION_DIR/rviz.pid"

echo "RViz is running in VM display $RVIZ_DISPLAY."
echo "VNC is bound to the SSH tunnel at vnc://localhost:${REMOTE_VNC_PORT:-5901}."
echo "Logs: $SESSION_DIR"
echo
echo "Run the sim stack in another terminal:"
echo "  cd $REMOTE_SIM_REPO/lidar_line_sim"
echo "  ./Run_LIDAR_LINE_ROS_COURSE.command"
echo

wait "$rviz_pid"
REMOTE
ssh_pid=$!
wait "$ssh_pid"
