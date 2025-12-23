import EnhancedUIOverlay from './components/EnhancedUIOverlay';
import EnhancedThreeCanvas from './components/EnhancedThreeCanvas';

function App() {
  return (
    <>
      <EnhancedThreeCanvas {...props} />
      <EnhancedUIOverlay {...props} />
    </>
  );
}
