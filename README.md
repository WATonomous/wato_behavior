# wato_behavior
Behavior model for WATO ASD Stack

# Running on WATO server
1. Go (cd) into wato_behaviour repo
2. Run `docker build -t wato_behavior:latest .` if not built yet
3. Run `docker run --name wato_behaviour -d -v "$(pwd):/home/bolty/wato_behaviour" wato_behavior:latest`

# Dependency setup steps (for local setup)
Following steps are for command prompt
1. Create virtual environment (name it venv): `python -m venv venv`
2. Activate virtual environment: `venv\Scripts\activate.bat`
3. Install metadrive env: 
```
git clone -b main --single-branch https://github.com/metadriverse/metadrive.git  --depth 1
cd metadrive
pip install -e .
```
4. Go back to project directory: `cd ..`
5. Pull metadrive assets with `python -m metadrive.pull_asset --update`
6. Install other dependencies: `pip install -r requirements.txt`

# Tasks to do 
- Improve occupancy grid emulation (add naive temporal element)
- Try implementing quasi-"Mixture Of Experts" model
- Think of random ideas to try
