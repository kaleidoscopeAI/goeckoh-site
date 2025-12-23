async function sendQuery() {
  const text = document.getElementById('input').value;
  const response = await fetch('/think', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text})
  });
  const data = await response.json();
  document.getElementById('output').innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
