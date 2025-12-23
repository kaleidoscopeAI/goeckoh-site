async def root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html><body><h1>Final AGI</h1>
    <div id="root"></div>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script type="text/babel">
        const App = () => <div>Groundbreaking AGI - Phi: {Math.random().toFixed(2)} | History: {Math.floor(Math.random()*1000)}</div>;
        ReactDOM.render(<App />, document.getElementById('root'));
    </script></body></html>
    """)

