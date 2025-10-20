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
params = {'a': 1.5, 'b': 1.0, 'c': 0.0, 'phase': 0.0, 't': 0.0, 'r': 1.0}
x_min, x_max = -6.28, 6.28
y_min, y_max = -3.0, 3.0

# Plotting Modes
PLOT_MODE_CARTESIAN = "cartesian"
PLOT_MODE_POLAR = "polar"
PLOT_MODE_PARAMETRIC = "parametric"
PLOT_MODE_3D = "3d"

plot_mode = PLOT_MODE_CARTESIAN

# UI States
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

# Example Expressions By Categories
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

# Color Pairs
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
                # Implicit: f(x,y) = 0 or f(x,y) = constant
                left_expr = sympify(left, locals=get_math_context())
                right_expr = sympify(right, locals=get_math_context())
                # Move everything to left side: f(x,y) - constant = 0
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
    """Evaluate y = f(x) - optimized"""
    x_sym = sp.Symbol('x')
    results = []
    param_subs = [(sp.Symbol(k), v) for k, v in params.items()]
    
    # Substitute parameters once
    expr_with_params = expr.subs(param_subs)
    
    # Try to use lambdify for speed (lagging earlier)
    try:
        func = sp.lambdify(x_sym, expr_with_params, 'math')
        for x_val in x_vals:
            try:
                y_val = func(x_val)
                if abs(y_val) < 1e6:  # Filter infinities
                    results.append((x_val, y_val))
            except (ValueError, ZeroDivisionError):
                # Handle domain errors (like sqrt of negative, division by zero)
                pass
            except:
                pass
    except:
        # Otherwise Fallback (To Slow Method)
        for x_val in x_vals:
            try:
                result = expr_with_params.subs(x_sym, x_val)
                result = float(result.evalf())
                if abs(result) < 1e6:
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
    """Evaluate implicit equation - optimized"""
    x_sym = sp.Symbol('x')
    y_sym = sp.Symbol('y')
    points = []
    
    # Get current parameter values and substitute them
    param_subs = [(sp.Symbol(k), v) for k, v in params.items()]
    
    try:
        # Substitute parameters into the expression
        expr_with_params = expr.subs(param_subs)
        
        # Converting to lambda for faster evaluation
        try:
            func = sp.lambdify((x_sym, y_sym), expr_with_params, 'math')
            use_lambda = True
        except:
            use_lambda = False
        
        # Sample a few points to determine scale for adaptive threshold
        test_vals = []
        for i in range(0, len(x_vals), max(1, len(x_vals)//10)):
            for j in range(0, len(y_vals), max(1, len(y_vals)//10)):
                try:
                    if use_lambda:
                        val = abs(func(x_vals[i], y_vals[j]))
                    else:
                        val = abs(float(expr_with_params.subs([(x_sym, x_vals[i]), (y_sym, y_vals[j])]).evalf()))
                    if val < 100:  # Filter extreme values
                        test_vals.append(val)
                except:
                    pass
        
        # Adaptive threshold based on value range
        if test_vals:
            threshold = max(0.1, min(test_vals) + 0.3)
        else:
            threshold = 0.5
        
        # Sample grid
        for x_val in x_vals:
            for y_val in y_vals:
                try:
                    if use_lambda:
                        result = float(func(x_val, y_val))
                    else:
                        result = expr_with_params.subs([(x_sym, x_val), (y_sym, y_val)])
                        result = float(result.evalf())
                    
                    if abs(result) < threshold:
                        points.append((x_val, y_val))
                except:
                    pass
    except:
        pass
    
    return points

def evaluate_parametric(x_expr, y_expr, t_vals):
    """Evaluate parametric equations - optimized"""
    t_sym = sp.Symbol('t')
    points = []
    param_subs = [(sp.Symbol(k), v) for k, v in params.items()]
    
    x_expr_param = x_expr.subs(param_subs)
    y_expr_param = y_expr.subs(param_subs)
    
    # Try lambdify for speed
    try:
        x_func = sp.lambdify(t_sym, x_expr_param, 'math')
        y_func = sp.lambdify(t_sym, y_expr_param, 'math')
        
        for t_val in t_vals:
            try:
                x_val = x_func(t_val)
                y_val = y_func(t_val)
                if abs(x_val) < 1e6 and abs(y_val) < 1e6:
                    points.append((x_val, y_val))
            except:
                pass
    except:
        # Fallback
        for t_val in t_vals:
            try:
                x_val = float(x_expr_param.subs(t_sym, t_val).evalf())
                y_val = float(y_expr_param.subs(t_sym, t_val).evalf())
                points.append((x_val, y_val))
            except:
                pass
    
    return points

def evaluate_polar(r_expr, t_vals):
    """Evaluate polar equation r = f(t) - optimized"""
    t_sym = sp.Symbol('t')
    points = []
    param_subs = [(sp.Symbol(k), v) for k, v in params.items()]
    r_expr_param = r_expr.subs(param_subs)
    
    # Try lambdify
    try:
        r_func = sp.lambdify(t_sym, r_expr_param, 'math')
        
        for t_val in t_vals:
            try:
                r_val = r_func(t_val)
                if abs(r_val) < 100:  # Reasonable bounds
                    x_val = r_val * math.cos(t_val)
                    y_val = r_val * math.sin(t_val)
                    points.append((x_val, y_val))
            except:
                pass
    except:
        # Fallback
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
    
    # Simple orthographic projection
    for x_val in x_vals:
        for y_val in y_vals:
            try:
                z_val = float(z_expr_param.subs([(x_sym, x_val), (y_sym, y_val)]).evalf())
                # Map z to color intensity
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
            x_vals = [x_min + (x_max - x_min) * i / (plot_width * 1.5)
                     for i in range(int(plot_width * 1.5))]
            points = evaluate_explicit_y(expr, x_vals)
        
        elif expr_type == "x_explicit":
            y_vals = [y_min + (y_max - y_min) * i / plot_height 
                     for i in range(plot_height * 2)]
            points = evaluate_explicit_x(expr, y_vals)
        
        elif expr_type == "implicit":
            # Reduced resolution for speed - adaptive based on zoom
            range_factor = (x_max - x_min) / 12.56  # Relative to default range
            samples_x = max(40, min(80, int(plot_width * 1.5 / range_factor)))
            samples_y = max(30, min(60, int(plot_height * 1.5 / range_factor)))
            
            x_vals = [x_min + (x_max - x_min) * i / samples_x 
                     for i in range(samples_x)]
            y_vals = [y_min + (y_max - y_min) * i / samples_y 
                     for i in range(samples_y)]
            points = evaluate_implicit(expr, x_vals, y_vals)
        
        elif expr_type == "parametric":
            x_expr, y_expr = expr
            t_vals = [i * 2 * math.pi / 150 for i in range(150)]  # Reduced from 200
            points = evaluate_parametric(x_expr, y_expr, t_vals)
        
        elif expr_type == "polar":
            t_vals = [i * 2 * math.pi / 200 for i in range(200)]  # Reduced from 300
            points = evaluate_polar(expr, t_vals)
        
        elif expr_type == "3d":
            # Reduced sampling for 3D
            samples_x = min(30, plot_width // 2)
            samples_y = min(20, plot_height // 2)
            x_vals = [x_min + (x_max - x_min) * i / samples_x
                     for i in range(samples_x)]
            y_vals = [y_min + (y_max - y_min) * i / samples_y
                     for i in range(samples_y)]
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

def find_intersections():
    """Find intersection points between first two expressions"""
    if len(expressions) < 2:
        return []
    
    try:
        expr1, type1, _ = parse_expression(expressions[0])
        expr2, type2, _ = parse_expression(expressions[1])
        
        if type1 == "y_explicit" and type2 == "y_explicit":
            x_sym = sp.Symbol('x')
            param_subs = [(sp.Symbol(k), v) for k, v in params.items()]
            
            eq = expr1.subs(param_subs) - expr2.subs(param_subs)
            solutions = solve(eq, x_sym)
            
            intersections = []
            for sol in solutions[:5]:  # Limit to 5 intersections
                try:
                    x_val = float(sol.evalf())
                    if x_min <= x_val <= x_max:
                        y_val = float(expr1.subs(param_subs).subs(x_sym, x_val).evalf())
                        if y_min <= y_val <= y_max:
                            intersections.append((x_val, y_val))
                except:
                    pass
            
            return intersections
    except:
        pass
    
    return []

# ============================================================================
# DRAWING FUNCTIONS
# ============================================================================

def draw_header(stdscr):
    """Draw header with title and status"""
    height, width = stdscr.getmaxyx()
    
    try:
        # Mode Indicators In Title
        mode_icons = {
            PLOT_MODE_CARTESIAN: "📊",
            PLOT_MODE_POLAR: "🔄",
            PLOT_MODE_PARAMETRIC: "〰️",
            PLOT_MODE_3D: "📦"
        }
        
        title = f"╔═══ GRAPHING CALC {mode_icons.get(plot_mode, '')} ═══╗"
        stdscr.addstr(0, max(0, (width - len(title)) // 2), title, 
                     curses.color_pair(COLOR_HEADER) | curses.A_BOLD)
        
        # Mode & Status
        if edit_mode:
            mode_str = "✎ EDIT"
            hint = "Type | ← → | Enter=done | Esc=cancel"
        elif animating:
            mode_str = "◉ ANIM"
            hint = "W=stop | Esc=quit"
        else:
            mode_str = "◆ NAV"
            hint = "Space=Edit | W=Anim | E=Examples | M=Mode | D=Deriv | I=Intersect | T=Table"
        
        status_line = f"[{mode_str}] {status_msg}"
        stdscr.addstr(1, 2, status_line[:width-4], 
                     curses.color_pair(COLOR_MODE) | curses.A_BOLD)
        
        if len(hint) < width - 4:
            stdscr.addstr(2, 2, hint[:width-4], curses.color_pair(COLOR_HEADER))
    except curses.error:
        pass

def draw_expressions(stdscr, start_row):
    """Draw expression list"""
    height, width = stdscr.getmaxyx()
    row = start_row
    
    try:
        header = f"EXPRESSIONS [{current_category.upper()}]: (E=examples, M=mode)"
        stdscr.addstr(row, 2, header[:width-4], 
                     curses.color_pair(COLOR_HEADER) | curses.A_BOLD)
        row += 1
        
        for i, expr in enumerate(expressions):
            if row >= height - 1:
                break
            
            is_active = (i == current_line)
            _, expr_type, err = parse_expression(expr)
            
            # Typing Icons
            type_icons = {
                "y_explicit": "↗",
                "x_explicit": "↑",
                "implicit": "◎",
                "parametric": "〰",
                "polar": "🔄",
                "3d": "📦"
            }
            type_icon = "⚠" if err else type_icons.get(expr_type, "?")
            
            color = COLOR_EXPR_ACTIVE if is_active else COLOR_EXPR
            prefix = "►" if is_active else " "
            
            display = expr
            if is_active and edit_mode:
                display = expr[:cursor_pos] + "┃" + expr[cursor_pos:]
                prefix = "✎"
            
            line = f" {prefix} {type_icon} {display}"
            if len(line) > width - 4:
                line = line[:width-7] + "..."
            
            attr = curses.color_pair(color)
            if is_active:
                attr |= curses.A_BOLD
            if err and expr.strip():
                attr = curses.color_pair(COLOR_ERROR)
            
            stdscr.addstr(row, 2, line, attr)
            row += 1
        
        return row + 1
    except curses.error:
        return row

def draw_params(stdscr, start_row):
    """Draw parameters"""
    if not show_params:
        return start_row
    
    height, width = stdscr.getmaxyx()
    row = start_row
    
    try:
        param_str = "PARAMS: " + " ".join([f"{k}={v:.2f}" for k, v in params.items()])
        stdscr.addstr(row, 2, param_str[:width-4], curses.color_pair(COLOR_PARAMS))
        row += 1
        return row
    except curses.error:
        return row

def draw_plot(stdscr, start_row):
    """Draw ASCII plot"""
    height, width = stdscr.getmaxyx()
    row = start_row
    
    try:
        plot_height = min(24, height - row - 6)
        plot_width = min(75, width - 4)
        
        plot_grid = [[' ' for _ in range(plot_width)] for _ in range(plot_height)]
        
        # Draw axes
        x_range_val = x_max - x_min
        y_range_val = y_max - y_min
        
        if x_min <= 0 <= x_max:
            zero_x_col = int((0 - x_min) / x_range_val * plot_width)
            zero_x_col = max(0, min(plot_width - 1, zero_x_col))
            for r in range(plot_height):
                plot_grid[r][zero_x_col] = '│'
        
        if y_min <= 0 <= y_max:
            zero_y_row = int((y_max - 0) / y_range_val * plot_height)
            zero_y_row = max(0, min(plot_height - 1, zero_y_row))
            for col in range(plot_width):
                if plot_grid[zero_y_row][col] == '│':
                    plot_grid[zero_y_row][col] = '┼'
                else:
                    plot_grid[zero_y_row][col] = '─'
        
        # Plot expressions with different colors
        markers = ['*', 'o', '+', 'x', '#', '@', '%', '&']
        
        for expr_idx, expr_str in enumerate(expressions):
            if not expr_str.strip():
                continue
            
            points, expr_type, error = get_plot_points(expr_str, plot_width, plot_height)
            
            if error or not points:
                continue
            
            marker = markers[expr_idx % len(markers)]
            
            for point_data in points:
                if expr_type == "3d" and len(point_data) == 3:
                    x_val, y_val, z_val = point_data
                    # Use z value to vary marker (simple shading)
                    if z_val > 0:
                        marker = '*'
                    else:
                        marker = 'o'
                else:
                    x_val, y_val = point_data[0], point_data[1]
                
                if x_min <= x_val <= x_max and y_min <= y_val <= y_max:
                    col = int((x_val - x_min) / x_range_val * plot_width)
                    row_plot = int((y_max - y_val) / y_range_val * plot_height)
                    
                    col = max(0, min(plot_width - 1, col))
                    row_plot = max(0, min(plot_height - 1, row_plot))
                    
                    if plot_grid[row_plot][col] not in ['┼']:
                        plot_grid[row_plot][col] = marker
        
        # Show intersections
        if show_intersections:
            intersections = find_intersections()
            for x_val, y_val in intersections:
                col = int((x_val - x_min) / x_range_val * plot_width)
                row_plot = int((y_max - y_val) / y_range_val * plot_height)
                
                col = max(0, min(plot_width - 1, col))
                row_plot = max(0, min(plot_height - 1, row_plot))
                plot_grid[row_plot][col] = '●'
        
        # Draw grid
        for r, line in enumerate(plot_grid):
            if row + r >= height - 2:
                break
            stdscr.addstr(row + r, 2, ''.join(line), curses.color_pair(COLOR_PLOT1))
        
        # Range info
        range_row = row + plot_height
        if range_row < height - 2:
            range_info = f"x:[{x_min:.1f},{x_max:.1f}] y:[{y_min:.1f},{y_max:.1f}] +/-=Zoom L/R=Pan"
            stdscr.addstr(range_row, 2, range_info[:width-4], curses.color_pair(COLOR_HEADER))
        
        return range_row + 1
    except curses.error:
        return row

def draw_derivatives(stdscr, start_row):
    """Draw derivative information"""
    if not show_derivatives or not expressions[current_line].strip():
        return start_row
    
    height, width = stdscr.getmaxyx()
    row = start_row
    
    try:
        stdscr.addstr(row, 2, "DERIVATIVE:", 
                     curses.color_pair(COLOR_HEADER) | curses.A_BOLD)
        row += 1
        
        deriv = compute_derivative(expressions[current_line])
        if deriv:
            deriv_str = f"  f'(x) = {str(deriv)}"
            stdscr.addstr(row, 2, deriv_str[:width-4], curses.color_pair(COLOR_DERIV))
            row += 1
        else:
            stdscr.addstr(row, 2, "  Cannot compute derivative", 
                         curses.color_pair(COLOR_ERROR))
            row += 1
        
        return row + 1
    except curses.error:
        return row

def draw_intersections_info(stdscr, start_row):
    """Draw intersection information"""
    if not show_intersections or len(expressions) < 2:
        return start_row
    
    height, width = stdscr.getmaxyx()
    row = start_row
    
    try:
        stdscr.addstr(row, 2, "INTERSECTIONS:", 
                     curses.color_pair(COLOR_HEADER) | curses.A_BOLD)
        row += 1
        
        intersections = find_intersections()
        if intersections:
            for x_val, y_val in intersections[:3]:
                info = f"  ● ({x_val:.3f}, {y_val:.3f})"
                stdscr.addstr(row, 2, info[:width-4], curses.color_pair(COLOR_MODE))
                row += 1
        else:
            stdscr.addstr(row, 2, "  No intersections found", 
                         curses.color_pair(COLOR_ERROR))
            row += 1
        
        return row + 1
    except curses.error:
        return row

def draw_table(stdscr, start_row):
    """Draw values table"""
    if not show_table:
        return start_row
    
    height, width = stdscr.getmaxyx()
    row = start_row
    
    try:
        stdscr.addstr(row, 2, "TABLE:", 
                     curses.color_pair(COLOR_HEADER) | curses.A_BOLD)
        row += 1
        
        x_points = [x_min + (x_max - x_min) * i / 10 for i in range(11)]
        
        header = f"  {'x':<8} "
        for i in range(min(len(expressions), 3)):
            if expressions[i].strip():
                header += f"f{i+1:<8} "
        stdscr.addstr(row, 2, header[:width-4], curses.A_BOLD)
        row += 1
        
        for x_val in x_points:
            if row >= height - 2:
                break
            
            line = f"  {x_val:<8.2f} "
            
            for expr_str in expressions[:3]:
                if not expr_str.strip():
                    line += f"{'---':<8} "
                    continue
                
                expr, expr_type, err = parse_expression(expr_str)
                
                if err or expr is None or expr_type != "y_explicit":
                    line += f"{'N/A':<8} "
                else:
                    try:
                        x_sym = sp.Symbol('x')
                        param_subs = [(sp.Symbol(k), v) for k, v in params.items()]
                        result = expr.subs(param_subs)
                        y_val = float(result.subs(x_sym, x_val).evalf())
                        line += f"{y_val:<8.2f} "
                    except:
                        line += f"{'ERR':<8} "
            
            stdscr.addstr(row, 2, line[:width-4])
            row += 1
        
        return row + 1
    except curses.error:
        return row

def draw_footer(stdscr):
    """Draw footer"""
    height, width = stdscr.getmaxyx()
    
    try:
        if error_msg:
            stdscr.addstr(height - 1, 2, f"⚠ {error_msg}"[:width-4], 
                         curses.color_pair(COLOR_ERROR) | curses.A_BOLD)
        elif not edit_mode and not animating:
            tips = [
                "💡 Implicit: (x²+y²)²-a*(x²-y²)=0 (lemniscate) | x²+y²-phase=0 (expanding)",
                "💡 Functions: abs(sin(x+phase)) | exp(-a*x²)*cos(b*x+phase) - All animate!",
                "💡 C=categories | E=examples | I=intersect | D=derivative | M=modes",
            ]
            tip = tips[int(time.time() / 3) % len(tips)]
            stdscr.addstr(height - 1, 2, tip[:width-4], curses.color_pair(COLOR_HEADER))
    except curses.error:
        pass

def draw_screen(stdscr):
    """Main draw function - optimized to prevent flicker"""
    # Doesn't clear entire screen - uses erase instead
    stdscr.erase()  # Less aggressive than clear()
    
    row = 3
    draw_header(stdscr)
    
    row = draw_expressions(stdscr, row)
    row = draw_params(stdscr, row)
    row = draw_derivatives(stdscr, row)
    row = draw_intersections_info(stdscr, row)
    row = draw_plot(stdscr, row)
    row = draw_table(stdscr, row)
    
    draw_footer(stdscr)
    
    stdscr.refresh()

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
        # Change example category
        if not edit_mode:
            categories = list(EXAMPLE_EXPRESSIONS.keys())
            current_idx = categories.index(current_category)
            current_category = categories[(current_idx + 1) % len(categories)]
            current_example_index = 0
            
            # Helpful hints for each category
            hints = {
                "waves": "Try W to animate waves!",
                "polar": "Beautiful roses and spirals - animate with W!",
                "parametric": "Lissajous curves - press W!",
                "implicit": "Curves & shapes - W animates them!",
                "3d": "3D surfaces - animate with W!",
                "functions": "Classic functions"
            }
            status_msg = f"Category: {current_category.upper()} - {hints.get(current_category, 'Press E')} Press E for examples"
    
    elif key == ord('m') or key == ord('M'):
        # Cycle plot modes
        modes = [PLOT_MODE_CARTESIAN, PLOT_MODE_POLAR, PLOT_MODE_PARAMETRIC, PLOT_MODE_3D]
        current_idx = modes.index(plot_mode)
        plot_mode = modes[(current_idx + 1) % len(modes)]
        
        mode_names = {
            PLOT_MODE_CARTESIAN: "Cartesian (y=f(x))",
            PLOT_MODE_POLAR: "Polar (r=f(t))",
            PLOT_MODE_PARAMETRIC: "Parametric (x=f(t), y=g(t))",
            PLOT_MODE_3D: "3D (z=f(x,y))"
        }
        status_msg = f"Mode: {mode_names[plot_mode]}"
    
    elif key == ord('p') or key == ord('P'):
        show_params = not show_params
        status_msg = "Params " + ("ON" if show_params else "OFF")
    
    elif key == ord('t') or key == ord('T'):
        show_table = not show_table
        status_msg = "Table " + ("ON" if show_table else "OFF")
    
    elif key == ord('d') or key == ord('D'):
        show_derivatives = not show_derivatives
        status_msg = "Derivatives " + ("ON" if show_derivatives else "OFF")
    
    elif key == ord('i') or key == ord('I'):
        show_intersections = not show_intersections
        status_msg = "Intersections " + ("ON" if show_intersections else "OFF")
    
    elif key == ord('w') or key == ord('W'):
        animating = not animating
        if animating:
            status_msg = "🎬 Animating!"
        else:
            status_msg = "Stopped"
    
    elif key == ord('x') or key == ord('X'):
        # Delete current expression
        if len(expressions) > 1 and not edit_mode:
            del expressions[current_line]
            current_line = min(current_line, len(expressions) - 1)
            cursor_pos = 0
            status_msg = "Deleted"
    
    elif key == ord('0'):
        # Reset view
        global x_min, x_max, y_min, y_max
        x_min, x_max = -6.28, 6.28
        y_min, y_max = -2.5, 2.5
        status_msg = "View reset"

def animate_phase(stdscr):
    """Animate phase parameter - optimized"""
    global params, animating, status_msg
    
    stdscr.nodelay(True)
    status_msg = "🎬 Animating..."
    
    frame_count = 0
    last_frame_time = time.time()
    
    while animating:
        current_time = time.time()
        
        # Frame rate limiting (target 20 FPS = 0.05s per frame)
        if current_time - last_frame_time < anim_speed:
            time.sleep(0.01)  # Small sleep to prevent busy-waiting
            
            # Check for key press during sleep
            try:
                key = stdscr.getch()
                if key != -1 and (key == ord('w') or key == ord('W') or key == 27):
                    animating = False
                    status_msg = "Stopped"
                    params['a'] = 1.5
                    params['b'] = 1.0
                    break
            except:
                pass
            continue
        
        last_frame_time = current_time
        
        # Update parameters
        params['phase'] = (params['phase'] + 0.2) % (2 * math.pi)
        params['c'] = params['phase']
        params['t'] = params['t'] + 0.08
        
        # Vary parameters for animation effects
        params['a'] = 1.5 + 0.8 * math.sin(params['t'])
        params['b'] = 1.0 + 0.6 * math.cos(params['t'] * 0.7)
        params['r'] = 1.0 + 0.3 * math.cos(params['t'] * 0.5)
        
        # Redraw with erase instead of clear
        stdscr.erase()
        row = 3
        draw_header(stdscr)
        row = draw_expressions(stdscr, row)
        row = draw_params(stdscr, row)
        row = draw_plot(stdscr, row)
        draw_footer(stdscr)
        stdscr.refresh()
        
        frame_count += 1
        
        # Check for interrupt
        try:
            key = stdscr.getch()
            if key != -1:
                if key == ord('w') or key == ord('W') or key == 27:
                    animating = False
                    status_msg = "Stopped"
                    params['a'] = 1.5
                    params['b'] = 1.0
                    break
        except:
            pass
    
    stdscr.nodelay(False)

# ============================================================================
# MAIN LOOP
# ============================================================================

def init_colors():
    """Initialize color pairs"""
    curses.start_color()
    curses.use_default_colors()
    
    # Enable double buffering if available
    try:
        curses.curs_set(0)
    except:
        pass
    
    curses.init_pair(COLOR_HEADER, curses.COLOR_CYAN, -1)
    curses.init_pair(COLOR_EXPR, curses.COLOR_WHITE, -1)
    curses.init_pair(COLOR_EXPR_ACTIVE, curses.COLOR_YELLOW, -1)
    curses.init_pair(COLOR_PARAMS, curses.COLOR_GREEN, -1)
    curses.init_pair(COLOR_ERROR, curses.COLOR_RED, -1)
    curses.init_pair(COLOR_PLOT1, curses.COLOR_MAGENTA, -1)
    curses.init_pair(COLOR_PLOT2, curses.COLOR_BLUE, -1)
    curses.init_pair(COLOR_PLOT3, curses.COLOR_CYAN, -1)
    curses.init_pair(COLOR_MODE, curses.COLOR_GREEN, -1)
    curses.init_pair(COLOR_DERIV, curses.COLOR_YELLOW, -1)

def main(stdscr):
    """Main application loop"""
    global cursor_pos, status_msg, edit_mode, animating
    
    curses.curs_set(0)  # Hide cursor
    stdscr.keypad(True)  # Enable keypad mode
    curses.cbreak()      # React to keys without Enter
    stdscr.nodelay(False)  # Blocking mode by default
    
    init_colors()
    stdscr.clear()
    
    cursor_pos = len(expressions[current_line])
    status_msg = "Welcome! W=animate | E=examples | C=categories"
    
    # Draw initial screen
    draw_screen(stdscr)
    
    last_key_time = time.time()
    needs_redraw = False
    
    while True:
        # Only redraw if needed (reduces flickering)
        if needs_redraw and not animating:
            draw_screen(stdscr)
            needs_redraw = False
        
        if animating:
            animate_phase(stdscr)
            needs_redraw = True
            continue
        
        try:
            stdscr.timeout(100)  # 100ms timeout
            key = stdscr.getch()
            if key == -1:  # No key pressed
                continue
        except:
            continue
        
        # Process input
        if key == 27:  # ESC
            if edit_mode:
                edit_mode = False
                status_msg = "Edit cancelled"
                needs_redraw = True
            else:
                break
        elif key in [curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT]:
            handle_navigation(key)
            needs_redraw = True
        elif edit_mode and (32 <= key <= 126 or key in [curses.KEY_BACKSPACE, 127, 8, curses.KEY_DC]):
            handle_editing(key)
            needs_redraw = True
        elif key in [ord('+'), ord('='), ord('-'), ord('_'), ord('l'), ord('L'), ord('r'), ord('R')]:
            handle_zoom_pan(key)
            needs_redraw = True
        else:
            handle_special_commands(key)
            needs_redraw = True

def run():
    """Entry point"""
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run()
