from ui.dash_app import app
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%H:%M:%S')

if __name__ == "__main__":
    debug_mode = os.environ.get('DASH_DEBUG', 'False').lower() == 'true'
    
    #try:
    logging.info(f"Starting Dash server in {'debug' if debug_mode else 'production'} mode")
    app.run(
        debug=debug_mode,
        dev_tools_ui=debug_mode,
        dev_tools_hot_reload=debug_mode,
        dev_tools_serve_dev_bundles=debug_mode
    )
    #except SystemExit as e:
     #   if e.code == 3 and debug_mode:
     #       logging.info("Hot reload triggered - restarting server...")
     #       pass
     #   else:
     #       raise
    #except KeyboardInterrupt:
    #    logging.info("Received keyboard interrupt, shutting down server...")