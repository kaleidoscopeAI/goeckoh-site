def create_app():
    app = Flask(__name__)

    @app.route('/')
    def index():
        return TEMPLATE

    return app
