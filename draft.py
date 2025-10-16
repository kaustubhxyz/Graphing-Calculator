import curses
import time

# Global state (minimal for skeleton)
expressions = ["sin(x)", "cos(x)"]  # Placeholder exprs
plot_range = (-10, 10)
edit_mode = True
current_line = 0
cursor_pos = {'line': 0, 'col': 0}

# Layout constants
HEADER_HEIGHT = 4
PLOT_HEIGHT = 20
PLOT_WIDTH = 70

def clear_and_setup_screen(stdscr):
    curses.curs_set(1 if edit_mode else 0)
    stdscr.nodelay(False)
    stdscr.keypad(True)
    if curses.has_colors():
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    height, width = stdscr.getmaxyx()
    global PLOT_HEIGHT
    PLOT_HEIGHT = max(10, height - HEADER_HEIGHT - 5)

def draw_header(stdscr, width):
    title = "Terminal Desmos Clone v0.1 - Basic TUI Skeleton"
    stdscr.addstr(0, 0, title[:width], curses.A_BOLD | curses.A_REVERSE)
    status = f"Exprs: {len(expressions)} | Range: [{plot_range[0]:.1f}, {plot_range[1]:.1f}] | Edit Mode: {edit_mode}"
    stdscr.addstr(1, 0, status[:width], curses.A_BOLD)
    controls = "Arrows=Nav | Space=Edit Toggle | ESC=Quit (Skeleton - No Plots Yet)"
    stdscr.addstr(2, 0, controls[:width], curses.A_DIM)
    expr_label = "Expressions (Placeholder):"
    stdscr.addstr(3, 0, expr_label[:width], curses.A_BOLD)

def draw_expressions(stdscr, width):
    y_start = HEADER_HEIGHT
    for i, expr in enumerate(expressions):
        y = y_start + i
        attrs = curses.color_pair(3) | curses.A_BOLD if i == current_line else 0
        if edit_mode and i == current_line:
            display_expr = expr[:cursor_pos['col']] + "|" + expr[cursor_pos['col']:]
            stdscr.addstr(y, 2, display_expr[:width-10], attrs)
        else:
            stdscr.addstr(y, 2, expr[:width-10], attrs)

def draw_plot_area(stdscr, height, width):
    y_start = HEADER_HEIGHT + len(expressions) + 1
    plot_w = width - 5
    plot_h = min(PLOT_HEIGHT, height - y_start - 5)
    # Placeholder ASCII plot
    placeholder = ["Placeholder Plot Area (SymPy coming soon)"] * plot_h
    for i, line in enumerate(placeholder):
        y = y_start + i
        if y >= height: break
        stdscr.addstr(y, 0, line[:plot_w], curses.color_pair(1))

def draw_footer(stdscr, width):
    height, _ = stdscr.getmaxyx()
    footer_y = height - 2
    modes = f"Current Line: {current_line} | Cursor Col: {cursor_pos['col']}"
    stdscr.addstr(footer_y, 0, modes[:width], curses.A_DIM)
    quit_msg = "ESC to quit | Resize terminal and rerun"
    stdscr.addstr(footer_y + 1, 0, quit_msg[:width], curses.A_DIM)

def handle_keypress(stdscr, key):
    global current_line, cursor_pos, edit_mode
    if key == 27:  # ESC
        return False
    elif key == curses.KEY_UP:
        current_line = max(0, current_line - 1)
        cursor_pos['col'] = 0
    elif key == curses.KEY_DOWN:
        current_line = min(len(expressions) - 1, current_line + 1)
        cursor_pos['col'] = 0
    elif key == curses.KEY_LEFT and edit_mode:
        cursor_pos['col'] = max(0, cursor_pos['col'] - 1)
    elif key == curses.KEY_RIGHT and edit_mode:
        cursor_pos['col'] = min(len(expressions[current_line]), cursor_pos['col'] + 1)
    elif key == ord(' '):
        edit_mode = not edit_mode
        curses.curs_set(1 if edit_mode else 0)
    elif key == curses.KEY_RESIZE:
        stdscr.clear()
        return True
    return True

def main(stdscr):
    clear_and_setup_screen(stdscr)
    while True:
        height, width = stdscr.getmaxyx()
        draw_header(stdscr, width)
        draw_expressions(stdscr, width)
        draw_plot_area(stdscr, height, width)
        draw_footer(stdscr, width)
        stdscr.refresh()
        key = stdscr.getch()
        if not handle_keypress(stdscr, key):
            break
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()

if __name__ == "__main__":
    curses.wrapper(main)
