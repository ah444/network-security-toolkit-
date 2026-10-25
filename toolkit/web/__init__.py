"""The web trainer (v5.2 step 2.3d): ``ui/`` (the page, the command's console messages),
``controllers/`` (the Flask routes, the ``web`` command), ``services/`` (workspaces, jobs),
``data/`` (uploads, the server's socket). It uses ``toolkit.core`` and never imports
``toolkit.cli``; ``toolkit.start`` sends it the ``web`` command."""
