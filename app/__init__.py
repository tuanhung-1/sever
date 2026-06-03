from __future__ import annotations


def create_app():
    from app.main import create_app as runtime_create_app

    return runtime_create_app()


__all__ = ["create_app"]
