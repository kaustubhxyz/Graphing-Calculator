#!/usr/bin/env python3
"""
Terminal Graphing Calculator - Advanced Edition
Supports: Parametric equations, polar coordinates, 3D projections, derivatives, integrals
Multi-color plots, equation solving, intersection finding, and more!
"""

import curses
import sys
import time
import io
from contextlib import redirect_stdout
import math

try:
    import sympy as sp
    from sympy.plotting.textplot import textplot
    from sympy import sympify, sin, cos, tan, sqrt, exp, log, pi, E, solve, diff, integrate
except ImportError:
    print("Error: sympy is required. Install with: pip install sympy")
    sys.exit(1)

# ============================================================================
# GLOBAL STATE
# ============================================================================

expressions = ["sin(x + phase) + 0.5*sin(2*x - phase)"]
params = {'a': 1.0, 'b': 1.0, 'c': 0.0, 'phase': 0.0, 't': 0.0, 'r': 1.0}
x_min, x_max = -6.28, 6.28
y_min, y_max = -2.5, 2.5

# Plotting modes
PLOT_MODE_CARTESIAN = "cartesian"
PLOT_MODE_POLAR = "polar"
PLOT_MODE_PARAMETRIC = "parametric"
PLOT_MODE_3D = "3d"

plot_mode = PLOT_MODE_CARTESIAN

# UI State
current_line = 0
edit_mode = False
show_params = True
show_table = False
show_derivatives = False
show_intersections = False
animating = False
error_msg = ""
status_msg = "Press M to change mode | W to animate | E for examples"

cursor_pos = 0
anim_speed = 0.03

# Enhanced example expressions by category
EXAMPLE_EXPRESSIONS = {
    "waves": [
        "sin(x + phase) + 0.5*sin(2*x - phase)",
        "exp(-0.1*x**2)*sin(x + phase)",
        "sin(3*x + phase) + 0.3*sin(7*x - 2*phase)",
    ],
    "polar": [
        "r=1 + sin(5*t + phase)",  # Rose curve
        "r=2*cos(4*t)",  # 4-petal rose
        "r=exp(t/10)",  # Spiral
        "r=1 + 0.5*sin(t + phase)",  # Animated circle
    ],
    "parametric": [
        "x=cos(t), y=sin(t)",  # Circle
        "x=t*cos(t), y=t*sin(t)",  # Spiral
        "x=cos(3*t), y=sin(5*t)",  # Lissajous
        "x=sin(t + phase), y=cos(2*t + phase)",  # Animated Lissajous
    ],
    "implicit": [
        "x**2 + y**2 = 4",  # Circle
        "(x**2 + y**2)**2 = a*(x**2 - y**2)",  # Lemniscate (animated with a)
        "x**2/a + y**2/b = 4",  # Ellipse (animated)
        "y**2 = x**3 - x",  # Elliptic curve
        "x**2 - y**2 = a",  # Hyperbola (animated)
        "(x**2 + y**2 - a)**2 = b*(x**2 - y**2)",  # Complex curve
        "sin(x)*cos(y) = a/4",  # Wave grid (animated)
        "x**2 + y**2 - phase = 0",  # Expanding circle
    ],
    "3d": [
        "z=sin(sqrt(x**2 + y**2) + phase)",  # Ripple
        "z=exp(-0.1*(x**2 + y**2))*cos(x + phase)",  # Gaussian wave
        "z=sin(x + phase)*cos(y)",  # Saddle wave
    ],
    "functions": [
        "abs(sin(x))",
        "1/(1 + exp(-x))",  # Sigmoid
        "x*sin(1/x)",  # Pathological
        "floor(x) + 0.5*sin(pi*x)",  # Step function
    ]
}

current_category = "waves"
current_example_index = 0

# Color pairs
COLOR_HEADER = 1
COLOR_EXPR = 2
COLOR_EXPR_ACTIVE = 3
COLOR_PARAMS = 4
COLOR_ERROR = 5
COLOR_PLOT1 = 6
COLOR_PLOT2 = 7
COLOR_PLOT3 = 8
COLOR_MODE = 9
COLOR_DERIV = 10

# ============================================================================
# EXPRESSION PARSING & EVALUATION
# ============================================================================

def get_math_context():
    """Return dictionary of available math functions for sympify"""
    return {
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'sqrt': sp.sqrt, 'exp': sp.exp, 'log': sp.log,
        'pi': sp.pi, 'e': sp.E, 'abs': sp.Abs,
        'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
        'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
        'floor': sp.floor, 'ceiling': sp.ceiling,
        **params
    }

def parse_expression(expr_str):
    """Parse expression and determine type"""
    try:
        expr_str = expr_str.strip()
        if not expr_str:
            return None, "empty", None
        
        expr_str = expr_str.replace('^', '**')
        expr_str = expr_str.replace('π', 'pi')
        
        x_sym = sp.Symbol('x')
        y_sym = sp.Symbol('y')
        t_sym = sp.Symbol('t')
        r_sym = sp.Symbol('r')
        z_sym = sp.Symbol('z')
        
        # Parametric: x=f(t), y=g(t)
        if ',' in expr_str and ('x=' in expr_str or 'y=' in expr_str):
            parts = expr_str.split(',')
            x_part = parts[0].strip()
            y_part = parts[1].strip() if len(parts) > 1 else ""
            
            if x_part.startswith('x=') and y_part.startswith('y='):
                x_expr = sympify(x_part[2:], locals=get_math_context())
                y_expr = sympify(y_part[2:], locals=get_math_context())
                return (x_expr, y_expr), "parametric", None
        
        # Polar: r = f(t) or r = f(theta)
        if expr_str.startswith('r='):
            r_expr = sympify(expr_str[2:], locals=get_math_context())
            return r_expr, "polar", None
        
        # 3D: z = f(x,y)
        if expr_str.startswith('z='):
            z_expr = sympify(expr_str[2:], locals=get_math_context())
            return z_expr, "3d", None
        
        # Explicit y = f(x) or x = f(y)
        if '=' in expr_str and expr_str.count('=') == 1:
            left, right = expr_str.split('=')
            left = left.strip()
            right = right.strip()
            
            if left == 'y':
                expr = sympify(right, locals=get_math_context())
                return expr, "y_explicit", None
            elif left == 'x':
                expr = sympify(right, locals=get_math_context())
                return expr, "x_explicit", None
            else:
                left_expr = sympify(left, locals=get_math_context())
                right_expr = sympify(right, locals=get_math_context())
                implicit_expr = left_expr - right_expr
                return implicit_expr, "implicit", None
        
        # No equals - assume y = f(x)
        else:
            expr = sympify(expr_str, locals=get_math_context())
            if y_sym in expr.free_symbols:
                return expr, "implicit", None
            else:
                return expr, "y_explicit", None
        
    except Exception as e:
        return None, "error", str(e)

def evaluate_explicit_y(expr, x_vals):
    """Evaluate y = f(x)"""
    x_sym = sp.Symbol('x')
    results = []
    param_subs = [(sp.Symbol(k), v) for k, v in params.items()]
    
    for x_val in x_vals:
        try:
            result = expr.subs(param_subs)
            result = result.subs(x_sym, x_val)
            result = float(result.evalf())
            results.append((x_val, result))
        except:
            pass
    
    return results

def evaluate_explicit_x(expr, y_vals):
    """Evaluate x = f(y)"""
    y_sym = sp.Symbol('y')
    results = []
    param_subs = [(sp.Symbol(k), v) for k, v in params.items()]
    
    for y_val in y_vals:
        try:
            result = expr.subs(param_subs)
            result = result.subs(y_sym, y_val)
            result = float(result.evalf())
            results.append((result, y_val))
        except:
            pass
    
    return results

def evaluate_implicit(expr, x_vals, y_vals):
    """Evaluate implicit equation"""
    x_sym = sp.Symbol('x')
    y_sym = sp.Symbol('y')
    points = []
    
    # Get current parameter values and substitute them
    param_subs = [(sp.Symbol(k), v) for k, v in params.items()]
    
    try:
        # Substitute parameters into the expression
        expr_with_params = expr.subs(param_subs)
        
        # Sample a grid and find where f(x,y) ≈ 0
        for x_val in x_vals:
            for y_val in y_vals:
                try:
                    result = expr_with_params.subs([(x_sym, x_val), (y_sym, y_val)])
                    result = float(result.evalf())
                    
                    # If close to zero, it's on the curve
                    # Use adaptive threshold based on range
                    threshold = 0.3
                    if abs(result) < threshold:
                        points.append((x_val, y_val))
                except:
                    pass
    except:
        pass
    
    return points

def evaluate_parametric(x_expr, y_expr, t_vals):
    """Evaluate parametric equations"""
    t_sym = sp.Symbol('t')
    points = []
    param_subs = [(sp.Symbol(k), v) for k, v in params.items()]
    
    x_expr_param = x_expr.subs(param_subs)
    y_expr_param = y_expr.subs(param_subs)
    
    for t_val in t_vals:
        try:
            x_val = float(x_expr_param.subs(t_sym, t_val).evalf())
            y_val = float(y_expr_param.subs(t_sym, t_val).evalf())
            points.append((x_val, y_val))
        except:
            pass
    
    return points

def evaluate_polar(r_expr, t_vals):
    """Evaluate polar equation r = f(t)"""
    t_sym = sp.Symbol('t')
    points = []
    param_subs = [(sp.Symbol(k), v) for k, v in params.items()]
    r_expr_param = r_expr.subs(param_subs)
    
    for t_val in t_vals:
        try:
            r_val = float(r_expr_param.subs(t_sym, t_val).evalf())
            x_val = r_val * math.cos(t_val)
            y_val = r_val * math.sin(t_val)
            points.append((x_val, y_val))
        except:
            pass
    
    return points

def evaluate_3d(z_expr, x_vals, y_vals):
    """Evaluate 3D function and project to 2D"""
    x_sym = sp.Symbol('x')
    y_sym = sp.Symbol('y')
    points = []
    param_subs = [(sp.Symbol(k), v) for k, v in params.items()]
    z_expr_param = z_expr.subs(param_subs)
    
    # Simple orthographic projection (ignore z for display)
    for x_val in x_vals:
        for y_val in y_vals:
            try:
                z_val = float(z_expr_param.subs([(x_sym, x_val), (y_sym, y_val)]).evalf())
                # Map z to color intensity (not position)
                if abs(z_val) < 10:  # Filter extreme values
                    points.append((x_val, y_val, z_val))
            except:
                pass
    
    return points

def get_plot_points(expr_str, plot_width, plot_height):
    """Get points to plot for any expression type"""
    expr, expr_type, error = parse_expression(expr_str)
    
    if error:
        return [], expr_type, error
    
    if expr_type == "empty":
        return [], expr_type, None
    
    points = []
    
    try:
        if expr_type == "y_explicit":
            x_vals = [x_min + (x_max - x_min) * i / plot_width 
                     for i in range(plot_width * 2)]
            points = evaluate_explicit_y(expr, x_vals)
        
        elif expr_type == "x_explicit":
            y_vals = [y_min + (y_max - y_min) * i / plot_height 
                     for i in range(plot_height * 2)]
            points = evaluate_explicit_x(expr, y_vals)
        
        elif expr_type == "implicit":
            # Higher resolution grid for implicit equations
            samples_x = plot_width * 3
            samples_y = plot_height * 3
            x_vals = [x_min + (x_max - x_min) * i / samples_x 
                     for i in range(samples_x)]
            y_vals = [y_min + (y_max - y_min) * i / samples_y 
                     for i in range(samples_y)]
            points = evaluate_implicit(expr, x_vals, y_vals)
        
        elif expr_type == "parametric":
            x_expr, y_expr = expr
            t_vals = [i * 2 * math.pi / 200 for i in range(200)]
            points = evaluate_parametric(x_expr, y_expr, t_vals)
        
        elif expr_type == "polar":
            t_vals = [i * 2 * math.pi / 300 for i in range(300)]
            points = evaluate_polar(expr, t_vals)
        
        elif expr_type == "3d":
            x_vals = [x_min + (x_max - x_min) * i / (plot_width) 
                     for i in range(plot_width)]
            y_vals = [y_min + (y_max - y_min) * i / (plot_height) 
                     for i in range(plot_height)]
            points = evaluate_3d(expr, x_vals, y_vals)
        
        return points, expr_type, None
    
    except Exception as e:
        return [], expr_type, str(e)

def compute_derivative(expr_str):
    """Compute derivative of expression"""
    try:
        expr, expr_type, error = parse_expression(expr_str)
        if error or expr_type != "y_explicit":
            return None
        
        x_sym = sp.Symbol('x')
        derivative = diff(expr, x_sym)
        return derivative
    except:
        return None
# ============================================================================
# INPUT HANDLING
# ============================================================================

def handle_navigation(key):
    """Handle arrow key navigation"""
    global current_line, cursor_pos
    
    if not edit_mode:
        if key == curses.KEY_UP:
            current_line = max(0, current_line - 1)
            cursor_pos = len(expressions[current_line])
        elif key == curses.KEY_DOWN:
            current_line = min(len(expressions) - 1, current_line + 1)
            cursor_pos = len(expressions[current_line])
    else:
        if key == curses.KEY_LEFT:
            cursor_pos = max(0, cursor_pos - 1)
        elif key == curses.KEY_RIGHT:
            cursor_pos = min(len(expressions[current_line]), cursor_pos + 1)

def handle_editing(key):
    """Handle text editing"""
    global cursor_pos, expressions, error_msg, status_msg
    
    expr = expressions[current_line]
    
    if key == curses.KEY_BACKSPACE or key == 127 or key == 8:
        if cursor_pos > 0:
            expressions[current_line] = expr[:cursor_pos-1] + expr[cursor_pos:]
            cursor_pos -= 1
            error_msg = ""
            status_msg = "Editing..."
    elif key == curses.KEY_DC:
        if cursor_pos < len(expr):
            expressions[current_line] = expr[:cursor_pos] + expr[cursor_pos+1:]
            error_msg = ""
    elif 32 <= key <= 126:
        char = chr(key)
        expressions[current_line] = expr[:cursor_pos] + char + expr[cursor_pos:]
        cursor_pos += 1
        error_msg = ""
        status_msg = "Editing..."

def handle_zoom_pan(key):
    """Handle zoom and pan"""
    global x_min, x_max, y_min, y_max, status_msg
    
    if key == ord('+') or key == ord('='):
        x_center = (x_min + x_max) / 2
        x_range = (x_max - x_min) * 0.4
        x_min = x_center - x_range
        x_max = x_center + x_range
        
        y_center = (y_min + y_max) / 2
        y_range = (y_max - y_min) * 0.4
        y_min = y_center - y_range
        y_max = y_center + y_range
        
        status_msg = "Zoomed in"
    elif key == ord('-') or key == ord('_'):
        x_center = (x_min + x_max) / 2
        x_range = (x_max - x_min) * 1.25
        x_min = x_center - x_range
        x_max = x_center + x_range
        
        y_center = (y_min + y_max) / 2
        y_range = (y_max - y_min) * 1.25
        y_min = y_center - y_range
        y_max = y_center + y_range
        
        status_msg = "Zoomed out"
    elif key == ord('l') or key == ord('L'):
        shift = (x_max - x_min) * 0.2
        x_min -= shift
        x_max -= shift
        status_msg = "Panned left"
    elif key == ord('r') or key == ord('R'):
        shift = (x_max - x_min) * 0.2
        x_min += shift
        x_max += shift
        status_msg = "Panned right"

def handle_special_commands(key):
    """Handle special commands"""
    global edit_mode, show_params, show_table, cursor_pos, expressions, current_line
    global animating, status_msg, error_msg, current_example_index, current_category
    global show_derivatives, show_intersections, plot_mode
    
    if key == ord(' '):
        edit_mode = not edit_mode
        if edit_mode:
            cursor_pos = len(expressions[current_line])
            status_msg = "Editing - type away!"
            error_msg = ""
        else:
            _, _, err = parse_expression(expressions[current_line])
            if err:
                error_msg = err
                status_msg = "Invalid! Press Space to fix"
            else:
                status_msg = "Saved ✓"
    
    elif key == ord('\n') or key == 10:
        if edit_mode:
            _, _, err = parse_expression(expressions[current_line])
            if err:
                error_msg = err
                status_msg = "Fix errors first"
            else:
                edit_mode = False
                error_msg = ""
                status_msg = "Done! Press E for examples"
        else:
            if len(expressions) < 10:
                expressions.append("")
                current_line = len(expressions) - 1
                cursor_pos = 0
                edit_mode = True
                status_msg = "New line!"
    
    elif key == ord('e') or key == ord('E'):
        if not edit_mode:
            # Cycle through examples in current category
            examples = EXAMPLE_EXPRESSIONS[current_category]
            current_example_index = (current_example_index + 1) % len(examples)
            expressions[current_line] = examples[current_example_index]
            cursor_pos = len(expressions[current_line])
            status_msg = f"{current_category.upper()} example {current_example_index + 1}/{len(examples)} - Try W to animate!"
            error_msg = ""
    
    elif key == ord('c') or key == ord('C'):
