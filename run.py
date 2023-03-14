

from main import app as app1
from werkzeug.serving import run_simple  # werkzeug development server


def create_and_run_app():
    run_simple('localhost', 5000, app1, use_reloader=True,
               use_debugger=True, use_evalex=True)


if __name__ == "__main__":
    app = create_and_run_app()
