import os

from waitress import serve

from rollscope.http import create_app


if __name__ == "__main__":
    application = create_app()
    serve(
        application,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "18945")),
        threads=4,
        max_request_body_size=52 * 1024 * 1024,
    )
