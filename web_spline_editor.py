# math
from numpy import *  # einer hiervon kann vielleicht weg
import numpy as np  # einer hiervon kann vielleicht weg
from matplotlib.pyplot import plot, show, gca, clf, annotate, subplots
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import comb
# for save and load
import pickle
from os.path import exists
from os import listdir, get_terminal_size
from flask import Flask, render_template, request, send_file
import ctypes
import math
matplotlib.use('Agg')

# custom
from colored import *
from manuals import *
app = Flask(__name__)


DEFAULT_RESOLUTION = 20
DEFAULT_DEFAULT_SPEED = 6.0
DEFAULT_CHAIKIN_SMOOTHNESS = 6
DEFAULT_CHAIKIN_RATIO = 1/4

all_points = {}
all_curves = {}
next_p_id = 0
changed = False
auto_update = True
resolution = DEFAULT_RESOLUTION
chaikin_smoothness = DEFAULT_CHAIKIN_SMOOTHNESS
default_speed = DEFAULT_DEFAULT_SPEED



# TODO edit chaikins ratio # ? für alle, oder für jeden einzeln ?

# TODO errormessage >_edit curve [name] remove [nr] --> funktioniert, gibt aber einen Fehler aus
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/process', methods=['POST'])
def process():
    global errorMessage
   
    try:
        
        errorMessage=""
        input_text = request.form['input_text']
        image_path=programm(input_text)
        return errorMessage
       
            
        
            
        
        

        
       
    
    except Exception as e:
        # Log the exception
        print(e)
        # Handle the exception and return an appropriate response
        return "An error occurred", 500
# # Classes
class Curve:
    points = []
    color = "b"
    spline = "linear"
    show_point_connections = False
    display_label = False

    def __init__(self, point_s, s="linear", c="b", spc=False, sl=False):
        self.points = point_s
        self.spline = s
        self.color = c
        self.show_point_connections = spc
        self.display_label = sl

    # Functions
    def plot(self):
        global all_points
        x, y = [], []
        for p in self.points:
            x.append(all_points[p].get_x())
            y.append(all_points[p].get_y())

        plot_curve(x, y, self.spline, self.color)
        if self.show_point_connections:
            plot_curve(x, y, "linear", self.color, [2, 4])

    # Functions
    def animate(self, speed):
        global all_points
        x, y = [], []
        for p in self.points:
            x.append(all_points[p].get_x())
            y.append(all_points[p].get_y())
        animate(x, y, self.spline, self.color, speed)

    # Adder
    def add_point(self, point):
        self.points.append(point)

    def add_points(self, point_s):
        for p in point_s:
            self.points.append(p)

    # Modifier
    def change_spline(self, new_spline):
        self.spline = new_spline

    def change_color(self, new_color):
        self.color = new_color

    def change_point(self, nr, x, y):
        all_points[self.points[nr]].edit(x, y)

    def move_point(self, nr, pos):
        tmp = self.points[nr]
        self.points.remove(tmp)
        self.points[pos:pos] = [tmp]

    def switch_points(self, nr1, nr2):
        self.points[nr1], self.points[nr2] = self.points[nr2], self.points[nr1]

    def connect_points(self, state):
        self.show_point_connections = state

    def show_label(self, state):
        self.display_label = state

    def show_label_points(self, state):
        for p in self.points:
            all_points[p].show_label(state)

    def replace_point(self, old, new):
        print("old:", self.points)
        if old in self.points:
            print("old inside")
            self.points[self.points.index(old)] = new
        print("new:", self.points)

    def replace_point_nr(self, nr, new):
        self.points[nr] = new

    def reverse_points(self):
        self.points.reverse()

    # Subtractor
    def remove_point(self, point):
        while point in self.points:
            self.points.remove(point)

    def remove_point_nr(self, pos):
        self.points.remove(self.points[pos])

    # Getter
    def get_points(self):
        return self.points

    def get_points_unique(self):
        return list(set(self.points))

    def get_spline(self):
        return self.spline

    def get_color(self):
        return self.color

    def add_label(self):
        return self.display_label

    # Transformer
    def __str__(self) -> str:
        global all_points
        string = self.spline + " "
        string += str(len(self.points))
        i = 0
        for p in self.points:
            string += " " + str(i) + "(" + all_points[p].__short_str__() + ")"
            i += 1
        return string

    def __str_p_names__(self) -> str:
        global all_points
        string = self.spline + " "
        string += str(len(self.points))
        for p in self.points:
            string += " " + p + "(" + all_points[p].__short_str__() + ")"
        return string

    def __save__(self):
        return {
            "p": self.points,
            "s": self.spline,
            "c": self.color,
            "spc": self.show_point_connections,
            "l": self.display_label
        }


class Point:
    x, y = 0.0, 0.0
    color = "r."
    marker = ""
    display_label = False

    def __init__(self, init_x=0.0, init_y=0.0, c="r.", sl=False, mark=""):
        self.x = float(init_x)
        self.y = float(init_y)
        self.color = c
        self.display_label = sl
        self.marker = mark

    # Functions
    def plot(self):
        if self.marker != "":
            plot(self.x, self.y, self.color, marker=self.marker)
        else:
            plot(self.x, self.y, self.color)

    # Modifiers
    def edit(self, new_x, new_y):
        self.x = new_x
        self.y = new_y

    def change_color(self, c):
        self.color = c

    def change_marker(self, m):
        self.marker = m

    def show_label(self, state):
        self.display_label = state

    def move_x_y(self, _x, _y):
        self.x += _x
        self.y += _y

    # Getter
    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_pos(self):
        return [self.x, self.y]

    def get_color(self):
        return self.color

    def get_marker(self):
        return self.marker

    def get_color_clean(self):
        if self.color[-1:] in [".", "*"]:
            return self.color[:-1]
        else:
            return self.color

    def add_label(self):
        return self.display_label

    # Other
    def __str__(self) -> str:
        return f"x={self.x}, y={self.y}"

    def __short_str__(self) -> str:
        return f"{self.x}/{self.y}"

    def __save__(self):
        return {
            "x": self.x,
            "y": self.y,
            "c": self.color,
            "m": self.marker,
            "l": self.display_label
        }


# # Animation
def animate(x, y, _spline, c, speed) -> None:
    spline = _spline.split("_")
    global changed
    if spline[0] == "bezier":
        if len(spline) == 1 or spline[1] == "n":
            animate_bezier(x, y, c, speed)
        else:
            old_x, old_y = [], []
            n = int(spline[1])
            i = 0
            sub_speed = speed / (len(x) / (int(spline[1]) - 1))
            if n > 1:
                while i + n - 1 < len(x):
                    tmp_x, tmp_y = animate_bezier(x[i:i + n], y[i:i + n], c, sub_speed, x, y, old_x, old_y)
                    for tx in tmp_x:
                        old_x.append(tx)
                    for ty in tmp_y:
                        old_y.append(ty)
                    i += n - 1
        changed = True
    elif spline[0] == "chaikin":
        if len(spline) == 1:
            end = "short"
        else:
            end = {"s": "short", "l": "long", "c": "closed", "c+": "closed+"}[spline[1]]
        animate_chaikin(x, y, c, speed, end)
        changed = True
    else:
        exclamation("can only animate Bezier and Chaikin")


def animate_chaikin(px, py, c, speed, ends) -> (list, list):
    if ends == "closed+":
        ends = "closed"
        if px[len(px)-1] == px[0] and py[len(py)-1] == py[0]:
            px = px[:-1]
            py = py[:-1]
    step_x, step_y = px, py
    pause_time = speed / chaikin_smoothness / 3
    for i in range(0, chaikin_smoothness):
        clf()
        plot(px, py, 'k', dashes=[2, 4])
        plot(px, py, 'k.')
        # if ends == "closed":
        #     plot([px[len(px) - 1], px[0]], [py[len(py) - 1], py[0]], 'k', dashes=[2, 4])
        plot(step_x, step_y, str(0.5), marker='.')
        plot(step_x, step_y, str(0.5))
        gca().set_aspect('equal', adjustable='box')
        plt.pause(pause_time)

        step_x, step_y = chaikin_curve_step(step_x, step_y, DEFAULT_CHAIKIN_RATIO, ends)
        plot(step_x, step_y, c, marker=".")
        gca().set_aspect('equal', adjustable='box')
        plt.pause(pause_time)
        plot(step_x, step_y, c)
        gca().set_aspect('equal', adjustable='box')
        plt.pause(pause_time)


def animate_bezier(px, py, c, speed, all_x=None, all_y=None, past_curve_x=None, past_curve_y=None) -> (list, list):
    if not all_x: all_x = px[:]
    if not all_y: all_y = py[:]
    x, y = [], []

    for t in frange(float(0), float(1) + float(1 / resolution), float(1 / resolution)):
        clf()
        plot(all_x, all_y, 'k', dashes=[2, 4])
        plot(all_x, all_y, 'k.')
        if past_curve_y and past_curve_y:
            plot(past_curve_x, past_curve_y, c)

        sx = [px[:]]
        sy = [py[:]]
        while len(sx[len(sx)-1]) > 1:
            sx.append([])
            sy.append([])
            current = len(sx)-2
            for i in range(len(sx[current])-1):
                new_x, new_y = get_between(sx[current][i], sy[current][i], sx[current][i+1], sy[current][i+1], t)
                sx[len(sx) - 1].append(new_x)
                sy[len(sx) - 1].append(new_y)

        step = 0.8 / (len(sx) - 1)
        current_gray = 0.0
        for i in range(len(sx)):
            plot(sx[i], sy[i], str(current_gray))
            plot(sx[i], sy[i], str(current_gray), marker='.')
            current_gray += step

        x.append(sx[len(sx)-1][0])
        y.append(sy[len(sy)-1][0])

        plot(x, y, c)
        plot(sx[len(sx)-1], sy[len(sy)-1], 'r*')

        gca().set_aspect('equal', adjustable='box')
        plt.pause(speed/(resolution+1))
    return x, y


def get_between(x1, y1, x2, y2, t) -> (float, float):
    return x1+(x2-x1)*t, y1+(y2-y1)*t


def frange(x, y, jump):  # Iterable
    while x < y:
        yield float(x)
        x = x + jump


# # Curve-Calculating
# # # Bezier
def bernstein_poly(i, n, t) -> float:
    return comb(n, i) * (t ** i) * (1 - t) ** (n - i)


def bezier_curve(x_points, y_points) -> (float, float):
    n_points = len(x_points)
    t = np.linspace(0.0, 1.0, resolution)
    polynomial_array = np.array(
        [bernstein_poly(i, n_points - 1, t) for i in range(0, n_points)])
    x_vals = np.dot(x_points, polynomial_array)
    y_vals = np.dot(y_points, polynomial_array)
    return x_vals, y_vals


# # # Chaikin
def qi_ri(p0, p1, ratio) -> (float, float):
    # reference -> https://www.cs.unc.edu/~dm/UNC/COMP258/LECTURES/Chaikins-Algorithm.pdf
    qi = (1 - ratio) * p0 + ratio * p1
    ri = ratio * p0 + (1 - ratio) * p1
    return qi, ri


def chaikin_curve(smoothness, x_points, y_points, ratio, ends="closed") -> (float, float):
    for i in range(0, smoothness):
        x_points, y_points = chaikin_curve_step(x_points, y_points, ratio, ends)
    if ends == "closed":
        x_points.append(x_points[0])
        y_points.append(y_points[0])
    return x_points, y_points


def chaikin_curve_step(x_points, y_points, ratio, ends) -> (list, list):
    x_segment = []
    y_segment = []
    if ends == "long":
        x_segment.append(x_points[0])
        y_segment.append(y_points[0])
    for k in range(0, len(x_points) - 1):
        qi_x, ri_x = qi_ri(x_points[k], x_points[k + 1], ratio)
        qi_y, ri_y = qi_ri(y_points[k], y_points[k + 1], ratio)
        x_segment.append(qi_x)
        x_segment.append(ri_x)
        y_segment.append(qi_y)
        y_segment.append(ri_y)
    if ends == "long":
        x_segment.append(x_points[len(x_points) - 1])
        y_segment.append(y_points[len(y_points) - 1])
    elif ends == "closed":
        qi_x, ri_x = qi_ri(x_points[len(x_points) - 1], x_points[0], ratio)
        qi_y, ri_y = qi_ri(y_points[len(y_points) - 1], y_points[0], ratio)
        x_segment.append(qi_x)
        x_segment.append(ri_x)
        y_segment.append(qi_y)
        y_segment.append(ri_y)
    return x_segment, y_segment


# # Plotting
def plot_curve(t, f, spline, c, dash=None) -> None:
    spline = spline.split("_")
    match spline[0]:
        case "linear":
            if not dash:
                plot(t, f, c)
            else:
                plot(t, f, c, dashes=dash)
        case "bezier":
            if len(spline) == 1 or spline[1] == "n":
                x_vals, y_vals = bezier_curve(t, f)
                plot(x_vals, y_vals, c)
            else:
                n = int(spline[1])
                i = 0
                if n == 2:
                    plot(t, f, c)
                elif n > 1:
                    while i + n - 1 < len(t):
                        x_vals, y_vals = bezier_curve(t[i:i + n], f[i:i + n])
                        plot(x_vals, y_vals, c)
                        i += n - 1
        case "chaikin":
            ratio = DEFAULT_CHAIKIN_RATIO
            if len(spline) == 1 or spline[1] == "s":
                x_vals, y_vals = chaikin_curve(chaikin_smoothness, t, f, ratio, "short")
                plot(x_vals, y_vals, c)
            elif spline[1] == "l":
                x_vals, y_vals = chaikin_curve(chaikin_smoothness, t, f, ratio, "long")
                plot(x_vals, y_vals, c)
            elif spline[1] == "c":
                x_vals, y_vals = chaikin_curve(chaikin_smoothness, t, f, ratio, "closed")
                plot(x_vals, y_vals, c)
            elif spline[1] == "c+":
                if t[len(t)-1] == t[0] and f[len(f)-1] == f[0]:
                    del t[-1]
                    del f[-1]
                x_vals, y_vals = chaikin_curve(chaikin_smoothness, t, f, ratio, "closed")
                plot(x_vals, y_vals, c)
        case _:
            exclamation("spline '" + spline[0] + "' not supported")


def plot_all_points() -> None:
    for p in all_points:
        if all_points[p].add_label():
            x = all_points[p].get_x()
            y = all_points[p].get_y()
            annotate(p, xy=(x, y), xytext=(x + .1, y + .1), color=all_points[p].get_color_clean())
        all_points[p].plot()


def plot_all_curves() -> None:
    for c in all_curves:
        if all_curves[c].add_label():
            point = all_points[all_curves[c].get_points()[0]]
            x = point.get_x()
            y = point.get_y()
            annotate(c, xy=(x, y), xytext=(x + .1, y - .3), color=all_curves[c].get_color())
        all_curves[c].plot()


# # Other
def add_to_all(name, to_add) -> None:
    if type(to_add) is Point:
        all_points[name] = to_add
    elif type(to_add) is Curve:
        all_curves[name] = to_add


def next_p_name() -> str:
    global next_p_id
    name = "p_" + str(next_p_id)
    next_p_id += 1
    return name


def save(name) -> None:
    good_to_go = False
    if exists(name + ".pkl"):
        if input("... overwrite " + name + "? (y/n): ") == "y":
            good_to_go = True
    else:
        good_to_go = True

    if good_to_go:
        alle = {"points": {}, "curves": {}, "settings": {}}
        for point in all_points:
            alle["points"][point] = all_points[point].__save__()
        for curve in all_curves:
            alle["curves"][curve] = all_curves[curve].__save__()
        alle["settings"]["res"] = resolution
        alle["settings"]["speed"] = default_speed
        alle["settings"]["npid"] = next_p_id
        alle["settings"]["smooth"] = chaikin_smoothness
        with open(name + '.pkl', 'wb') as file:
            pickle.dump(alle, file)


def load(name) -> None:
    global resolution, next_p_id, changed, all_points, all_curves, default_speed, chaikin_smoothness
    good_to_go = False
    if len(all_points) > 0 or len(all_curves) > 0:
        if input("... delete current elements? (y/n): ") == "y":
            good_to_go = True
    else:
        good_to_go = True

    if good_to_go:
        all_points = {}
        all_curves = {}
        with open(name + '.pkl', 'rb') as file:
            alle = pickle.load(file)
        resolution = alle["settings"]["res"]
        default_speed = alle["settings"]["speed"]
        next_p_id = alle["settings"]["npid"]
        if "smooth" in alle["settings"]:  # Temporary check
            chaikin_smoothness = alle["settings"]["smooth"]
        for point in alle["points"]:
            p = alle["points"][point]
            if "m" in p:  # Temporary check
                add_to_all(point, Point(p["x"], p["y"], p["c"], p["l"], p["m"]))
            else:
                add_to_all(point, Point(p["x"], p["y"], p["c"], p["l"]))
        for curve in alle["curves"]:
            c = alle["curves"][curve]
            add_to_all(curve, Curve(c["p"], c["s"], c["c"], c["spc"], c["l"]))
        changed = True


def load_add(name) -> None:
    global resolution, next_p_id, changed

    p_trans = {}
    with open(name + '.pkl', 'rb') as file:
        alle = pickle.load(file)

    if alle["settings"]["npid"] > next_p_id:
        next_p_id = alle["settings"]["npid"]

    for point in alle["points"]:
        add_on = 0
        new_point = point
        while new_point in all_points:
            add_on += 1
            new_point = point + "_" + str(add_on)
        p_trans[point] = new_point
        p = alle["points"][point]
        if "m" in p:  # Temporary check
            add_to_all(new_point, Point(p["x"], p["y"], p["c"], p["l"], p["m"]))
        else:
            add_to_all(new_point, Point(p["x"], p["y"], p["c"], p["l"]))

    for curve in alle["curves"]:
        add_on = 0
        new_curve = curve
        while new_curve in all_curves:
            add_on += 1
            new_curve = curve + "_" + str(add_on)
        trans_points = []
        for point in alle["curves"][curve]["p"]:
            trans_points.append(p_trans[point])
        c = alle["curves"][curve]
        add_to_all(new_curve, Curve(trans_points, c["s"], c["c"], c["spc"], c["l"]))

    changed = True


# # printing help
def print_command_correction(typ, w) -> None:
    if typ not in mans:
        exclamation("command not found", True)
        return
    commands = get_only_commands(typ)
    for i in range(len(commands)):
        if i >= len(w):
            print_possible_commands(typ, commands, w)
            break
        coms = check_command_for_word(commands, w[i], i)
        if len(coms) == 0:
            print_possible_commands(typ, commands, w)
            break
        else:
            commands = coms


def print_possible_commands(typ, commands, w) -> None:
    exclamation("command faulty", True)
    print("   did you mean one of these? :")
    for c in commands:
        print("     >_" + typ + " " + " ".join(c))
    exclamation("for further help check >_manual " + typ)


def check_command_for_word(commands, word, nr) -> list:
    coms = commands[:]
    for i in range(len(coms)-1, -1, -1):
        if len(coms[i]) > nr:  # TODO # diese Zeile ist nur ein Hotfix
            if coms[i][nr][0] != "[" and coms[i][nr] != word:
                coms.pop(i)
    return coms


def get_only_commands(typ) -> list:
    commands = []
    for line in mans[typ]:
        if line[0] == "c":
            commands.append(line[1].split()[1:])
    return commands


def exclamation(txt, all_red=False):
    global errorMessage
    errorMessage=txt
    if all_red:
        print(colored(" ! " + txt, "RED"))
    else:
        print(colored(" ! ", "RED") + txt)
    

    
    
    


def cnf(curve, all_red=False) -> None:
    exclamation("curve '" + curve + "' not found", all_red)


def pnf(point, all_red=False) -> None:
    exclamation("point '" + point + "' not found", all_red)


# # Handle input
def h_i_create(w) -> None:
    global next_p_id, changed
    if w[0] == "point":
        if len(w) != 4:
            return print_command_correction("create", w)
        add_to_all(w[1], Point(w[2], w[3]))
        changed = True
    elif w[0] == "curve":
        if len(w) == 2:
            add_to_all(w[1], Curve([]))
        elif w[2] == "with":
            points = []
            for i in range(3, len(w)):
                points.append(w[i])
            add_to_all(w[1], Curve(points))
            changed = True
    elif w[0] == "shape":
        extras = w[2:]
        name = None
        x, y = 0, 0
        r = 1
        while len(extras) > 1:
            match extras[0]:
                case "named":
                    name = extras[1]
                    extras = extras[2:]
                case "at":
                    x = float(extras[1])
                    y = float(extras[2])
                    extras = extras[3:]
                case "radius":
                    r = float(extras[1])
                    extras = extras[2:]
                case _: break
        if name is not None:
            if name in all_curves:
                return exclamation("curve '" + name + "' already exists")
        if w[1] == "bezier_circle":
            create_bezier_circle(x, y, r, name)
        else:
            shape_arr = w[1].split("-")
            if len(shape_arr) == 2:
                if shape_arr[1] != "gon": return exclamation("unknown shape '" + w[1] + "'")
                n = int(shape_arr[0])
                if n < 3: return exclamation("for n-gons, n must be at least 3")
                create_n_gon(n, x, y, r, name)
            elif len(shape_arr) == 3:
                if shape_arr[2] != "star": return exclamation("unknown shape '" + w[1] + "'")
                n = int(shape_arr[0])
                star = int(shape_arr[1])
                if n < 3: return exclamation("for stars, n must be at least 5")
                create_n_gon(n, x, y, r, name, star)
            else:
                return print_command_correction("create", w)
    else:
        return print_command_correction("create", w)


# creates a circle of radius r at position x y made of 4 bezier_4 segments
def create_bezier_circle(x=0.0, y=0.0, r=1.0, name=None) -> None:
    global changed

    # point creation
    a = r
    b = r * 0.55
    new_points = []
    for p in [[0, a], [b, a], [a, b], [a, 0], [a, -b], [b, -a], [0, -a], [-b, -a], [-a, -b], [-a, 0], [-a, b], [-b, a]]:
        p_name = next_p_name()
        add_to_all(p_name, Point(p[0]+x, p[1]+y))
        new_points.append(p_name)
    new_points.append(new_points[0])

    # name finding
    if name is None:
        i = 1
        new_name = "circle"
        while new_name in all_curves:
            i += 1
            new_name = "circle_" + str(i)
        name = new_name

    # curve creation
    add_to_all(name, Curve(new_points, s="bezier_4"))
    changed = True


# create an n-sided shape of radius r at position x y
#  it is a regular polygon that can be a star shape
def create_n_gon(n, x=0.0, y=0.0, r=1.0, name=None, star=1) -> None:
    if n < 3: return exclamation("n should be at least 3")
    if star < 1: return exclamation("star should be at least 1")
    n = int(n)
    star = int(star)
    global changed

    # checking
    if star > 1:
        if star >= n/2: return exclamation("second value must be smaller")
        if star % 2 == 0 and n % 2 == 0: return exclamation("if both values are even, the result is multiple shapes")
        if n % star == 0: return exclamation("if the first is a multiple of the second, the result is multiple shapes")
        if n / star == math.floor(n / star): return exclamation("n / star should not be int")

    # point creation
    angle_step = 2 * pi / n
    now_angle = 0
    new_points = []
    for i in range(n):
        px, py = rotate(0, r, 0, 0, now_angle)
        now_angle += angle_step
        p_name = next_p_name()
        add_to_all(p_name, Point(px + x, py + y))
        new_points.append(p_name)

    # starification
    if star > 1:
        points = new_points[:]
        new_points = []
        for i in range(n):
            new_points.append(points[(i * star) % n])
        if len(set(new_points)) != len(new_points):
            for p in points:
                all_points.pop(p)
            return exclamation("star {" + str(n) + "/" + str(star) + "} is faulty but i don't know how to filter that out")  # TODO oben schon filtern

    # name finding
    if name is None:
        i = 1
        if star > 1: new_name = str(n) + "-" + str(star) + "-star"
        else: new_name = str(n) + "-gon"
        while new_name in all_curves:
            i += 1
            if star > 1: new_name = str(n) + "-" + str(star) + "-star_" + str(i)
            else: new_name = str(n) + "-gon_" + str(i)
        name = new_name

    # curve creation
    new_points.append(new_points[0])
    add_to_all(name, Curve(new_points))
    changed = True


def h_i_list(w) -> None:
    if len(w) != 1 and len(w) != 2 and len(w) != 3 and len(w) != 5:
        return print_command_correction("list", w)
    match w[0]:
        case "points":
            if len(w) == 1:
                for p in all_points:
                    print(p + ": " + all_points[p].__str__())
            elif len(w) == 2:
                if w[1] == "table":
                    list_points_extended(False)
                elif w[1] == "extra":
                    list_points_extended(True)
                else: return print_command_correction("list", w)
            elif len(w) == 5:
                if w[2] != "sorted" or w[3] != "by": return print_command_correction("list", w)
                if w[1] == "table":
                    list_points_extended(False, sorted_by=w[4])
                elif w[1] == "extra":
                    list_points_extended(True, sorted_by=w[4])
                else: return print_command_correction("list", w)
            else: return print_command_correction("list", w)
        case "curves":
            if len(w) == 1:
                for c in all_curves:
                    print(c + ": " + all_curves[c].__str__())
            elif len(w) == 2:
                if w[1] == "table":
                    list_curves_extended(False)
                elif w[1] == "extra":
                    list_curves_extended(True)
                else: return print_command_correction("list", w)
            elif len(w) == 5:
                if w[2] != "sorted" or w[3] != "by": return print_command_correction("list", w)
                if w[1] == "table":
                    list_curves_extended(False, sorted_by=w[4])
                elif w[1] == "extra":
                    list_curves_extended(True, sorted_by=w[4])
                else: return print_command_correction("list", w)
            else: return print_command_correction("list", w)
        case "settings":
            print("resolution:", resolution)
            print("default speed:", default_speed)
            print("chaikin smoothness:", chaikin_smoothness)
        case "defaults":
            print("resolution:", DEFAULT_RESOLUTION)
            print("chaikin smoothness:", DEFAULT_CHAIKIN_SMOOTHNESS)
            print("chaikin ratio:", DEFAULT_CHAIKIN_RATIO)
            print("speed:", DEFAULT_DEFAULT_SPEED)
        case "splines":
            print("linear: direct point to point connection")
            print("bezier_n: bezier using all points (aka. 'bezier' without _n)")
            print("bezier_ + int>1: many connected beziers of specified order")
            print(" - bezier_4: cubic bezier")
            print(" - bezier_3: quadratic bezier")
            print(" - bezier_2: linear")
            print("chaikin: chaikin's curve with all points")
            print(" - chaikin_l (long) extends the end to the outer points")
            print(" - chaikin_c (closed) loops the end together")
            print(" - chaikin_c+ ignores last point, if it is the same as the first")
        case "saves":
            directory = listdir()
            for file in directory:
                if file[-4:] == ".pkl":
                    print(" - " + file[:-4])
        case "curve":
            if len(w) == 2:
                print(w[1] + ": " + all_curves[w[1]].__str__())
            elif w[2] == "names":
                print(w[1] + ": " + all_curves[w[1]].__str_p_names__())
            elif w[2] == "table":
                list_curve_extended(w[1])
            else:
                return print_command_correction("list", w)
        case "point":
            print(w[1] + ": " + all_points[w[1]].__str__())


def list_curve_extended(c) -> None:
    if c not in all_curves: return cnf(c)
    # l_ = Longest # _b = before point # _a = after point
    l_name = 4
    l_x_b = 1
    l_y_b = 1
    l_x_a = 0
    l_y_a = 0
    nr_of_points = len(all_curves[c].get_points())
    l_nr = len(str(nr_of_points - 1))
    if l_nr < 2: l_nr = 2
    for p in all_curves[c].get_points():
        if len(p) > l_name: l_name = len(p)
        tmp_x = str(all_points[p].get_x()).split('.')
        tmp_y = str(all_points[p].get_y()).split('.')
        if len(tmp_x[0]) > l_x_b: l_x_b = len(tmp_x[0])
        if len(tmp_y[0]) > l_y_b: l_y_b = len(tmp_y[0])
        if len(tmp_x[1]) > l_x_a: l_x_a = len(tmp_x[1])
        if len(tmp_y[1]) > l_y_a: l_y_a = len(tmp_y[1])

    total_length = 11 + sum([l_nr, l_name, l_x_b, l_y_b, l_x_a, l_y_a])

    spline_str = "Spline: " + all_curves[c].get_spline()
    points_str = "Points: " + str(nr_of_points)
    curve_str = " " + c + " "
    hr = f"+-{'':-^{l_nr}}-+-{'':-^{l_name}}-+-{'':-^{l_x_b + l_x_a + 1}}-+-{'':-^{l_y_b + l_y_a + 1}}-+"
    print(f"+-{curve_str:-^{total_length}}-+")
    print(f"| {spline_str:<{total_length}} |")
    print(f"| {points_str:<{total_length}} |")
    print(hr)
    print(f"| {'Nr':^{l_nr}} | {'Name':^{l_name}} | {'X':^{l_x_b + l_x_a + 1}} | {'Y':^{l_y_b + l_y_a + 1}} |")
    print(hr)

    nr = 0
    for p in all_curves[c].get_points():
        x = str(all_points[p].get_x()).split('.')
        y = str(all_points[p].get_y()).split('.')
        print(f"| {nr:<{l_nr}} | {p:<{l_name}} | {x[0]:>{l_x_b}}.{x[1]:<{l_x_a}} | {y[0]:>{l_y_b}}.{y[1]:<{l_y_a}} |")
        nr += 1
    print(hr)


def list_curves_extended(extra=False, sorted_by=None) -> None:
    l_name = 4
    l_spline = 6
    l_nr = 2
    l_points = 6
    c_point_amount = {}
    c_point_names = {}
    c_point_values = {}

    for c in all_curves:
        if len(c) > l_name: l_name = len(c)
        curve = all_curves[c]
        if len(curve.get_spline()) > l_spline: l_spline = len(curve.get_spline())
        c_point_names[c] = curve.get_points()
        c_point_amount[c] = str(len(c_point_names[c]))
        if len(c_point_amount[c]) > l_nr: l_nr = len(c_point_amount[c])
        points_name_str = ""
        if extra:
            for p in c_point_names[c]:
                points_name_str += p + "(" + all_points[p].__short_str__() + "), "
        else:
            for p in c_point_names[c]:
                points_name_str += p + ", "
        points_name_str = points_name_str[:-2]
        c_point_values[c] = points_name_str
        if len(points_name_str) > l_points: l_points = len(points_name_str)

    sub_total_length = 14 + sum([l_name, l_spline, l_nr])
    space_for_points = get_terminal_size().columns - sub_total_length + 1
    if l_points <= space_for_points:
        l_points_2 = l_points
    else:
        l_points_2 = space_for_points

    header = f"| {'Name':^{l_name}} | {'Spline':^{l_spline}} | {'nr':^{l_nr}} | {'Points':^{l_points_2}} |"
    hr = f"+-{'':-^{l_name}}-+-{'':-^{l_spline}}-+-{'':-^{l_nr}}-+-{'':-^{l_points_2}}-+"
    half_empty = f"| {'':^{l_name}} | {'':^{l_spline}} | {'':^{l_nr}} | "

    list_entries = []
    for c in all_curves:
        curve = all_curves[c]
        more_lines = []
        tmp_str = f"| {c:<{l_name}} | {curve.get_spline():<{l_spline}} | {c_point_amount[c]:<{l_nr}} |"
        points = c_point_values[c]
        if len(points) > space_for_points:
            tmp_str += f" {points[:space_for_points]:<{l_points_2}} |"
            points = points[space_for_points:]
            while len(points) > space_for_points:
                more_lines.append(half_empty + f"{points[:space_for_points]:<{l_points_2}} |")
                points = points[space_for_points:]
            if len(points) > 0:
                more_lines.append(half_empty + f"{points:<{l_points_2}} |")
        else:
            tmp_str += f" {points:<{l_points_2}} |"

        list_entries.append([tmp_str, more_lines, c, curve.get_spline(), c_point_amount[c]])

    if sorted_by is not None:  # TODO Sortieren von Curves funktioniert nicht
        match sorted_by:
            case "name":
                list_entries = sorted(list_entries, key=lambda l: l[2])  # not working?
                pos = header.find(" Name ") + 1
                header = header[:pos] + colored("Name", "GREEN") + header[pos + 4:]
            case "spline":
                list_entries = sorted(list_entries, key=lambda l: l[3])  # not working?
                pos = header.find(" Spline ") + 1
                header = header[:pos] + colored("Spline", "GREEN") + header[pos + 6:]
            case "nr":
                list_entries = sorted(list_entries, key=lambda l: l[4])  # not working?
                pos = header.find(" nr ") + 1
                header = header[:pos] + colored("nr", "GREEN") + header[pos + 2:]

    print(f"+-{'Curves':-^{sub_total_length + l_points_2 - 5}}-+")
    print(hr)
    print(header)
    print(hr)
    for line in list_entries:
        print(line[0])
        for extra_line in line[1]:
            print(extra_line)
    print(hr)


def list_points_extended(extra=False, sorted_by=None) -> None:
    # l_ = Longest # _b = before point # _a = after point
    l_name = 4
    l_x_b = 1
    l_y_b = 1
    l_x_a = 0
    l_y_a = 0
    for p in all_points:
        if len(p) > l_name: l_name = len(p)
        tmp_x = str(all_points[p].get_x()).split('.')
        tmp_y = str(all_points[p].get_y()).split('.')
        if len(tmp_x[0]) > l_x_b: l_x_b = len(tmp_x[0])
        if len(tmp_y[0]) > l_y_b: l_y_b = len(tmp_y[0])
        if len(tmp_x[1]) > l_x_a: l_x_a = len(tmp_x[1])
        if len(tmp_y[1]) > l_y_a: l_y_a = len(tmp_y[1])

    header = f"| {'Name':^{l_name}} | {'X':^{l_x_b + l_x_a + 1}} | {'Y':^{l_y_b + l_y_a + 1}} |"
    hr = f"+-{'':-^{l_name}}-+-{'':-^{l_x_b + l_x_a + 1}}-+-{'':-^{l_y_b + l_y_a + 1}}-+"

    p_curve_amount = {}
    p_curve_names = {}
    l_c_a = 2
    l_c_n = 6
    if extra:
        for p in all_points:
            for c in all_curves:
                if p in all_curves[c].get_points():
                    # add to dicts
                    if p in p_curve_amount: p_curve_amount[p] += 1
                    else: p_curve_amount[p] = 1
                    if p in p_curve_names: p_curve_names[p] += ", " + c
                    else: p_curve_names[p] = c
            if p not in p_curve_amount: p_curve_amount[p] = 0
            if p not in p_curve_names: p_curve_names[p] = ""
        for p in p_curve_amount:
            if len(str(p_curve_amount[p])) > l_c_a: l_c_a = len(str(p_curve_amount[p]))
        for p in p_curve_names:
            if len(p_curve_names[p]) > l_c_n: l_c_n = len(p_curve_names[p])
        total_length = 14 + sum([l_name, l_x_b, l_y_b, l_x_a, l_y_a, l_c_a, l_c_n])
        hr += f"-{'':-^{l_c_a}}-+-{'':-^{l_c_n}}-+"
        header += f" {'in':^{l_c_a}} | {'Curves':^{l_c_n}} |"
    else:
        total_length = 8 + sum([l_name, l_x_b, l_y_b, l_x_a, l_y_a])

    list_entries = []
    for p in all_points:
        x = str(all_points[p].get_x()).split('.')
        y = str(all_points[p].get_y()).split('.')
        tmp_str = f"| {p:<{l_name}} | {x[0]:>{l_x_b}}.{x[1]:<{l_x_a}} | {y[0]:>{l_y_b}}.{y[1]:<{l_y_a}} |"
        if extra:
            tmp_str += f" {p_curve_amount[p]:>{l_c_a}} | {p_curve_names[p]:<{l_c_n}} |"
            list_entries.append([tmp_str, p, all_points[p].get_x(), all_points[p].get_y(), p_curve_amount[p]])
        else:
            list_entries.append([tmp_str, p, all_points[p].get_x(), all_points[p].get_y()])

    if sorted_by is not None:
        match sorted_by:
            case "name":
                list_entries = sorted(list_entries, key=lambda l: l[1])
                pos = header.find(" Name ")+1
                header = header[:pos] + colored("Name", "GREEN") + header[pos+4:]
            case "x":
                list_entries = sorted(list_entries, key=lambda l: l[2])
                pos = header.find(" X ")+1
                header = header[:pos] + colored("X", "GREEN") + header[pos+1:]
            case "y":
                list_entries = sorted(list_entries, key=lambda l: l[3])
                pos = header.find(" Y ")+1
                header = header[:pos] + colored("Y", "GREEN") + header[pos+1:]
            case "in":
                if extra:
                    list_entries = sorted(list_entries, key=lambda l: l[4])
                    pos = header.find(" in ")+1
                    header = header[:pos] + colored("in", "GREEN") + header[pos+2:]

    print(f"+-{'Points':-^{total_length}}-+")
    print(hr)
    print(header)
    print(hr)
    for line in list_entries:
        print(line[0])
    print(hr)


def h_i_add(w) -> None:
    global next_p_id, changed
    if len(w) == 3:
        all_curves[w[2]].add_point(w[0])
        changed = True
    if len(w) == 4:
        add_to_all("p_" + str(next_p_id), Point(w[2], w[3]))
        all_curves[w[1]].add_point("p_" + str(next_p_id))
        next_p_id += 1
        changed = True
    else:
        return print_command_correction("add", w)


def h_i_edit(w) -> None:
    global changed
    if len(w) < 4: return print_command_correction("edit", w)
    if w[0] == "curve":
        curve = w[1]
        if curve not in all_curves: return cnf(curve)

        changed = True  # vorsorglich true
        match w[2]:
            case "spline":
                if len(w) != 4: return print_command_correction("edit", w)
                all_curves[curve].change_spline(w[3])
            case "color":
                if len(w) != 4: return print_command_correction("edit", w)
                all_curves[curve].change_color(w[3])
            case "point_connection":
                if len(w) != 4: return print_command_correction("edit", w)
                if w[3] == "true":
                    all_curves[curve].connect_points(True)
                elif w[3] == "false":
                    all_curves[curve].connect_points(False)
            case "point":
                if len(w) != 6: return print_command_correction("edit", w)
                all_curves[curve].change_point(int(w[3]), int(w[4]), int(w[5]))
            case "remove":  # TODO test
                if len(w) == 5 and w[3] == "point":
                    all_curves[w[1]].remove_point_nr(int(w[4]))
                    changed = True
                if len(w) == 7 and w[3] == "point":
                    to_remove = all_curves[w[1]].get_points()[int(w[4])]
                    exterminate_point(to_remove)
                    changed = True
                elif len(w) == 6 or len(w) == 4:
                    if w[3] not in all_points: return pnf(w[3])
                    all_curves[w[1]].remove_point(w[4])
                    if len == 6 and w[4] == "&" and w[5] == "delete":
                        exterminate_point(w[4])
                    changed = True
                else: return print_command_correction("edit", w)
            case "switch":
                if len(w) != 5: return print_command_correction("edit", w)
                all_curves[curve].switch_points(int(w[3]), int(w[4]))
            case "replace":
                print("---in edit replace")
                global next_p_id

                if len(w) == 7 and w[3] == "point" and w[5] == "with":
                    if w[6] not in all_points: return pnf(w[6])
                    all_curves[curve].replace_point_nr(int(w[4]), w[6])
                    changed = True

                if len(w) == 8 and w[3] == "point" and w[5] == "with":
                    new_name = "p_" + str(next_p_id)
                    add_to_all(new_name, Point(w[6], w[7]))
                    all_curves[w[1]].add_point(new_name)
                    all_curves[curve].replace_point_nr(int(w[4]), new_name)
                    next_p_id += 1
                    changed = True

                if len(w) == 6 and w[4] == "with":
                    if w[3] not in all_points: return pnf(w[3])
                    if w[5] not in all_points: return pnf(w[5])
                    all_curves[curve].replace_point(w[3], w[5])
                    changed = True

                if len(w) == 7 and w[4] == "with":
                    if w[3] not in all_points: return pnf(w[3])
                    new_name = "p_" + str(next_p_id)
                    add_to_all(new_name, Point(w[5], w[6]))
                    all_curves[w[1]].add_point(new_name)
                    all_curves[curve].replace_point(w[3], new_name)
                    next_p_id += 1
                    changed = True

            case "move":
                if len(w) != 7: return print_command_correction("edit", w)
                print("using", w[4], "and", w[6])
                all_curves[curve].move_point(int(w[4]), int(w[6]))
            case "reverse":
                if len(w) != 4 and w[3] != "points": return print_command_correction("edit", w)
                all_curves[curve].reverse_points()
            case _:
                changed = False  # doch nicht true
    elif w[0] == "point":
        if len(w) != 4: return print_command_correction("edit", w)
        if w[2] == "color":
            all_points[w[1]].change_color(w[3])
            changed = True
        elif w[2] == "marker":
            all_points[w[1]].change_marker(w[3])
            changed = True
        else:
            all_points[w[1]].edit(float(w[2]), float(w[3]))
            changed = True
    else: print_command_correction("edit", w)


def exterminate_point(to_remove) -> None:
    for c in all_curves:
        all_curves[c].remove_point(to_remove)
    all_points.pop(to_remove)


def h_i_change(w) -> None:
    global changed
    if len(w) != 2:
        return print_command_correction("change", w)
    match w[0]:
        case "resolution":
            global resolution
            resolution = int(w[1])
            changed = True
        case "smoothness":
            global chaikin_smoothness
            chaikin_smoothness = int(w[1])
            changed = True
        case "auto_update":
            global auto_update
            if w[1] == "false":
                auto_update = False
            elif w[1] == "true":
                auto_update = True
            else:
                return print_command_correction("change", w)
        case "speed":
            global default_speed
            if float(w[1]) <= 0.0:
                default_speed = 0.1
            else:
                default_speed = float(w[1])


def h_i_copy(w) -> None:
    global changed
    if len(w) != 3 and len(w) != 5: return print_command_correction("copy", w)

    if w[0] == "curve":
        if is_available_or_yessed("curve", w[2]):
            if len(w) == 5 and w[3] == "new" and w[4] == "points":
                old_points = all_curves[w[1]].get_points()
                points = []
                for p in old_points:
                    new_id = 1
                    while p + "_" + str(new_id) in all_points:
                        new_id += 1
                    name = p + "_" + str(new_id)
                    add_to_all(name, Point(all_points[p].get_x(), all_points[p].get_y(), all_points[p].get_color()))
                    points.append(name)
            else:
                points = all_curves[w[1]].get_points()
            add_to_all(w[2], Curve(points))
            all_curves[w[2]].change_spline(all_curves[w[1]].get_spline())
            all_curves[w[2]].change_color(all_curves[w[1]].get_color())
    elif w[0] == "point":
        if is_available_or_yessed("point", w[2]):
            add_to_all(w[2], Point(all_points[w[1]].get_x(), all_points[w[1]].get_y(), all_points[w[1]].get_color()))


def h_i_save(w) -> None:
    if len(w) != 2:
        print_command_correction("save", w)
        return
    if w[0] != "as":
        print_command_correction("save", w)
        return
    save(w[1])


def h_i_load(w) -> None:
    if len(w) != 2:
        print_command_correction("load", w)
        return
    if w[0] == "from":
        load(w[1])
    elif w[0] == "additional":
        load_add(w[1])
    else:
        print_command_correction("load", w)
        return


def h_i_delete(w) -> None:
    global changed
    if len(w) == 2:
        if w[0] == "point":
            point = w[1]
            if point not in all_points: return pnf(point)
            for c in all_curves:
                if point in all_curves[c].get_points():
                    all_curves[c].remove_point(point)
            all_points.pop(point)
            changed = True
        elif w[0] == "curve":
            h_i_delete_curve(w[1], False)
        elif w[0] == "unused" and w[1] == "points":
            points_to_delete = list(all_points.keys())
            h_i_delete_unused(points_to_delete)
        else: print_command_correction("delete", w)
    elif len(w) == 4:
        if w[0] == "curve" and w[2] == "&" and w[3] == "points":
            h_i_delete_curve(w[1], True)
        else: print_command_correction("delete", w)
    else: print_command_correction("delete", w)


def h_i_delete_curve(name, points=False) -> None:
    global changed
    if name not in all_curves:
        return cnf(name)
    if points:
        points_to_delete = all_curves[name].get_points()
        h_i_delete_unused(points_to_delete, name)
    all_curves.pop(name)
    changed = True


def h_i_delete_unused(points_to_delete, exclude=None) -> None:
    global changed
    for c in all_curves:
        if c != exclude:
            points_to_check = all_curves[c].get_points()
            for i in range(len(points_to_delete) - 1, -1, -1):
                if points_to_delete[i] in points_to_check:
                    points_to_delete.remove(points_to_delete[i])
    for p in points_to_delete:
        all_points.pop(p)
    changed = True


def h_i_label(w) -> None:
    if len(w) > 5 or len(w) < 3 or w[0] != "set":
        return print_command_correction("label", w)

    if w[1] == "hidden":  state = False
    elif w[1] == "visible": state = True
    else: return print_command_correction("label", w)

    global changed

    if len(w) == 3:
        if w[2] == "all":
            h_i_label_all("point", state)
            h_i_label_all("curve", state)
        else: return print_command_correction("label", w)
        changed = True
    elif len(w) == 4:
        if w[2] == "all":
            if w[3] == "points":
                h_i_label_all("point", state)
            elif w[3] == "curves":
                h_i_label_all("curve", state)
            else: return print_command_correction("label", w)
            changed = True
        elif w[2] == "point":
            if w[3] not in all_points: return pnf(w[3])
            all_points[w[3]].show_label(state)
            changed = True
        elif w[2] == "curve":
            if w[3] not in all_curves: return cnf(w[3])
            all_curves[w[3]].show_label(state)
            changed = True
        else: return print_command_correction("label", w)
    elif len(w) == 5:
        if w[2] != "points" or w[3] != "of": return print_command_correction("label", w)
        if w[4] not in all_curves: return cnf(w[4])
        all_curves[w[4]].show_label_points(state)
        changed = True


def h_i_label_all(typ, state) -> None:
    if typ == "point":
        to_use = all_points
    elif typ == "curve":
        to_use = all_curves
    else: return
    global changed

    for obj in to_use:
        to_use[obj].show_label(state)
    changed = True


def h_i_manual(w) -> None:
    if len(w) == 0:
        print(f"+------ Manual ------+")
        print(f"|                    |")
        print(f"| >_man [manual]     |")
        print(f"| >_manual [manual]  |")
        print(f"|                    |")
        print(f"| Available manuals: |")
        for m in mans:
            print(f"| - {m:<17}|")
        print(f"|                    |")
        print(f"+--------------------+")
    elif len(w) == 1:
        if w[0] not in mans:
            exclamation("unknown manual")
            return

        longest = 0
        for i in mans[w[0]]:
            if len(i) == 2:
                if len(i[1]) > longest:
                    longest = len(i[1])
        blank_line = f"| {'':<{longest}} |"
        for i in mans[w[0]]:
            match i[0]:
                case "c" | "l" | "n":
                    print(f"| {i[1]:<{longest}} |")
                case "bl":
                    print(blank_line)
                case "t":
                    print(f"+-{i[1]:-^{longest}}-+")
                    print(blank_line)
                case "h":
                    total_length = longest - len(i[1])
                    left_length = int(total_length / 2) + 1
                    right_length = int(total_length / 2) + 1
                    if total_length % 2 == 1:
                        right_length += 1
                    line = "+"
                    if left_length % 2 == 1:
                        line += " "
                    line = line + ("- " * int(left_length / 2))
                    line += i[1]
                    line = line + (" -" * int(right_length / 2))
                    if right_length % 2 == 1:
                        line += " "
                    line += "+"
                    print(line)
                    print(blank_line)
                case "not implemented":
                    print(f"! {'NOT IMPLEMENTED':^{longest}} !")
        print(blank_line)
        print(f"+-{'':-<{longest}}-+")


def h_i_animate(w) -> None:
    if len(w) != 1 and len(w) != 3: return print_command_correction("animate", w)
    if w[0] not in all_curves: return cnf(w[0])
    speed = default_speed
    if len(w) == 3 and w[1] == "speed":
        speed = float(w[2])
        if speed <= 0.0:
            speed = 0.1
    all_curves[w[0]].animate(speed)


def h_i_rename(w) -> None:
    if len(w) != 3: return print_command_correction("rename", w)
    if w[0] == "curve":
        if w[1] not in all_curves: return cnf(w[1])
        if is_available_or_yessed("curve", w[2]):
            all_curves[w[2]] = all_curves[w[1]]
            del all_curves[w[1]]
    elif w[0] == "point":
        if w[1] not in all_points: return pnf(w[1])
        if is_available_or_yessed("point", w[2]):
            for c in all_curves:
                all_curves[c].replace_point(w[1], w[2])
            all_points[w[2]] = all_points[w[1]]
            del all_points[w[1]]
    else:
        return print_command_correction("rename", w)


def h_i_modify(w) -> None:
    if len(w) < 4 or len(w) > 8:
        return print_command_correction("modify", w)
    global changed
    if w[0] == "point":
        if len(w) != 5 and len(w) != 8 and len(w) != 7: return print_command_correction("modify", w)
        point = w[1]
        if point not in all_points: return pnf(point)
        if w[2] == "move":
            if len(w) != 5: return print_command_correction("modify", w)
            all_points[point].move_x_y(float(w[3]), float(w[4]))
            changed = True
        if w[2] == "rotate":
            if len(w) != 8: return print_command_correction("modify", w)
            if w[3] != "around": return print_command_correction("modify", w)
            p = point
            if w[4] == "point":
                if w[5] not in all_points: return pnf(w[5])
                c_x = all_points[w[5]].get_x()
                c_y = all_points[w[5]].get_y()
            else:
                c_x = float(w[4])
                c_y = float(w[5])
            match w[6]:
                case "deg": rad = float(w[7]) * math.pi / 180
                case "pi":  rad = float(w[7]) * math.pi
                case "rad": rad = float(w[7])
                case _: return print_command_correction("modify", w)
            point = all_points[p]
            new_x, new_y = rotate(point.get_x(), point.get_y(), c_x, c_y, rad)
            all_points[p].edit(new_x, new_y)
            changed = True
        if w[2] == "scale":
            if len(w) != 7: return print_command_correction("modify", w)
            if w[4] != "from": return print_command_correction("modify", w)
            scale = float(w[3])
            if w[5] == "point":
                if w[6] not in all_points: return pnf(w[6])
                c_x = all_points[w[6]].get_x()
                c_y = all_points[w[6]].get_y()
            else:
                c_x = float(w[5])
                c_y = float(w[6])
            new_x, new_y = scale_point(all_points[point].get_x(), all_points[point].get_y(), c_x, c_y, scale)
            all_points[point].edit(new_x, new_y)
            changed = True
        else: return print_command_correction("modify", w)
    elif w[0] == "curve":
        curve = w[1]
        if curve not in all_curves: return cnf(curve)
        match w[2]:
            case "move":
                if len(w) != 5: return print_command_correction("modify", w)
                if is_unique_or_yessed(curve):
                    for p in all_curves[curve].get_points():
                        all_points[p].move_x_y(float(w[3]), float(w[4]))
                    changed = True
            case "rotate":
                if len(w) != 8: return print_command_correction("modify", w)
                if w[3] != "around": return print_command_correction("modify", w)
                if is_unique_or_yessed(curve):
                    if w[4] == "point":
                        if w[5] not in all_points: return pnf(w[5])
                        c_x = all_points[w[5]].get_x()
                        c_y = all_points[w[5]].get_y()
                    else:
                        c_x = float(w[4])
                        c_y = float(w[5])
                    match w[6]:
                        case "deg": rad = float(w[7]) * math.pi / 180
                        case "pi": rad = float(w[7]) * math.pi
                        case "rad": rad = float(w[7])
                        case _: return print_command_correction("modify", w)
                    for p in all_curves[curve].get_points_unique():
                        point = all_points[p]
                        new_x, new_y = rotate(point.get_x(), point.get_y(), c_x, c_y, rad)
                        all_points[p].edit(new_x, new_y)
                    changed = True
            case "scale":
                if len(w) != 7: return print_command_correction("modify", w)
                if w[4] != "from": return print_command_correction("modify", w)
                if is_unique_or_yessed(curve):
                    scale = float(w[3])
                    if w[5] == "point":
                        if w[6] not in all_points: return pnf(w[6])
                        c_x = all_points[w[6]].get_x()
                        c_y = all_points[w[6]].get_y()
                    else:
                        c_x = float(w[5])
                        c_y = float(w[6])
                    for p in all_curves[curve].get_points_unique():
                        point = all_points[p]
                        new_x, new_y = scale_point(point.get_x(), point.get_y(), c_x, c_y, scale)
                        all_points[p].edit(new_x, new_y)
                    changed = True
            case "integrate":
                if len(w) != 4: return print_command_correction("modify", w)
                second_curve = w[3]
                if second_curve not in all_curves: return cnf(second_curve)
                all_curves[curve].add_points(all_curves[second_curve].get_points())
                all_curves.pop(second_curve)
                changed = True
            case _:
                return print_command_correction("modify", w)
    else:
        return print_command_correction("modify", w)


def rotate(px, py, ox, oy, a) -> (float, float):
    # danke an https://stackoverflow.com/a/34374437/7622658
    qx = ox + math.cos(a) * (px - ox) - math.sin(a) * (py - oy)
    qy = oy + math.sin(a) * (px - ox) + math.cos(a) * (py - oy)
    return qx, qy


def scale_point(px, py, ox, oy, a) -> (float, float):
    qx = (px - ox) * a + ox
    qy = (py - oy) * a + oy
    return qx, qy


def h_i_reset(w):
    if len(w) != 1: return print_command_correction("reset", w)
    if w[0] != "settings": return print_command_correction("reset", w)
    global resolution, chaikin_smoothness, default_speed
    resolution = DEFAULT_RESOLUTION
    chaikin_smoothness = DEFAULT_CHAIKIN_SMOOTHNESS
    default_speed = DEFAULT_DEFAULT_SPEED


def point_unique_to(point, curve) -> bool:
    for c in all_curves:
        if c != curve:
            if point in all_curves[c].get_points():
                return False
    return True


def points_unique_to(points, curve) -> bool:
    for p in points:
        if not point_unique_to(p, curve):
            return False
    return True


def is_unique_or_yessed(curve) -> bool:
    if points_unique_to(all_curves[curve].get_points(), curve):
        return True
    else:
        if input("... curve shares points with others. Move anyway? (y/n): ") == "y":
            return True
    return False


def is_available_or_yessed(typ, name) -> bool:
    if typ == "curve":
        if name not in all_curves:
            return True
    if typ == "point":
        if name not in all_points:
            return True
    if input("... name '" + name + "' already exists. Overwrite? (y/n): ") == "y":
        return True
    return False


def handle_input(txt) -> None:
    commands = txt.split("\\")
    for c in commands:
        words = c.split()
        if len(words) > 1:
            match words[0]:
                case "create":
                    h_i_create(words[1:])
                case "list":
                    h_i_list(words[1:])
                case "add":
                    h_i_add(words[1:])
                case "edit":
                    h_i_edit(words[1:])
                case "change":
                    h_i_change(words[1:])
                case "copy":
                    h_i_copy(words[1:])
                case "delete":
                    h_i_delete(words[1:])
                case "label":
                    h_i_label(words[1:])
                case "save":
                    h_i_save(words[1:])
                case "load":
                    h_i_load(words[1:])
                case "man":
                    h_i_manual(words[1:])
                case "animate":
                    h_i_animate(words[1:])
                case "modify":
                    h_i_modify(words[1:])
                case "rename":
                    h_i_rename(words[1:])
                case _:
                    exclamation("command not found", True)
                    
        else:
            if words[0] in mans:
                exclamation("for help check >_manual " + words[0])
            else:
                exclamation("command not found", True)


def repaint() -> None:
    clf()
    plot_all_curves()
    plot_all_points()
    gca().set_aspect('equal', adjustable='box')
    show(block=False)

def programm(text):
    global fig_path
    global changed
    handle_input(text)

    clf()

   
    for c in all_curves:
        if all_curves[c].add_label():
            point = all_points[all_curves[c].get_points()[0]]
            x = point.get_x()
            y = point.get_y()
            annotate(c, xy=(x, y), xytext=(x + .1, y - .3), color=all_curves[c].get_color())
        all_curves[c].plot()

    for p in all_points:
        if all_points[p].add_label():
            x = all_points[p].get_x()
            y = all_points[p].get_y()
            annotate(p, xy=(x, y), xytext=(x + .1, y + .1), color=all_points[p].get_color_clean())
        all_points[p].plot()
    
    plt.gca().set_aspect('equal', adjustable='box')
    fig_path = "static/fig.png"  # Save the figure in a static folder
    plt.savefig(fig_path)
    
    changed = False
    return fig_path

    
    

    
# # main
if __name__ == "__main__":
    app.run(debug=True)

    
    
