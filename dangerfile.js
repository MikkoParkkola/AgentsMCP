// Danger configuration to enforce PR standards.
const { danger, fail } = require('danger');

if (!danger.github.pr.body || danger.github.pr.body.length < 20) {
  fail('Please provide a detailed PR description.');
}
