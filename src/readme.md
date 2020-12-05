
## Lidar simulation on 3D car model

## Install requirements

```bash
python -m pip install -r requirements.txt
```

# How to run

- Run with the default configurations

```bash
python models.py
```

- Possible arguments:
  - `--with` : width of the car
  - `--length` : length of the car
  - `--height` : height of the car
  - `--range` : range of the lidar sensor
  - `-hres` : horizontal resolution of the lidar sensor
  - `-vres` : vertical resolution of the lidar sensor

- Examples

```bash
python models.py --width 1.95 --length 4.75 --height 1.5
```
