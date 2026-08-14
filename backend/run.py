"""Development entrypoint.

Production: `gunicorn --workers 2 --threads 4 --timeout 180 "run:app"`
(threads matter — SSE responses hold a worker for the length of the answer).
"""

from app import create_app
from config import settings

app = create_app()

if __name__ == "__main__":
    app.run(
        host=settings.host,
        port=settings.port,
        debug=settings.debug,
        threaded=True,
        # The reloader would re-instantiate the embedder and reload the index
        # on every file save, which is slow and confusing.
        use_reloader=False,
    )
