const amb = process.env.AMBIENTE || 'local';
require('http').createServer((req, res) => res.end(`Ola do ambiente: ${amb}`)).listen(3000);
