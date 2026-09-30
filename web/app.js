const chat = document.getElementById('chat');
const form = document.getElementById('form');
const input = document.getElementById('input');

function addMessage(text, who) {
  const div = document.createElement('div');
  div.className = `msg ${who}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function ask(query) {
  addMessage(query, 'user');
  const res = await fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({query})
  });
  const data = await res.json();
  addMessage(data.answer, 'bot');
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) return;
  input.value = '';
  await ask(query);
});

document.querySelectorAll('[data-q]').forEach((button) => {
  button.addEventListener('click', () => ask(button.dataset.q));
});
