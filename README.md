# wato_behavior
Behavior model for WATO ASD Stack

# Running on WATO server
1. Go (cd) into wato_behaviour repo
2. Run `docker build -t wato_behavior:latest .` if not built yet
3. Run `docker run --name wato_behaviour -d -v "$(pwd):/home/bolty/wato_behaviour" wato_behavior:latest`

# Dependency setup steps (for local setup)
1. Create virtual environment (name it venv): `python -m venv venv`
2. Activate virtual environment: `venv\Scripts\activate.bat`
3. Install metadrive env: 
```
git clone -b main --single-branch https://github.com/metadriverse/metadrive.git  --depth 1
cd metadrive
pip install -e .
```
4. Go back to project directory: `cd ..`
4. Install other dependencies: `pip install -r requirements.txt`

# Tasks to do 
- Change action space to output a trajectory (spline? line?) instead of raw controls
- Implement traditional control algorithm (mpc? pid?) to follow a set path for possible some kind of on-policy imitation learning/behaviour cloning
- Think of random ideas to try
