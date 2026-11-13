const amb = process.env.AMBIENTE || 'local';
require('http').createServer((req, res) => {
  console.log(`[${amb}] ${new Date().toISOString()} ${req.method} ${req.url}`);
  res.end(`Ola do ambiente: ${amb}`);
}).listen(3000);
