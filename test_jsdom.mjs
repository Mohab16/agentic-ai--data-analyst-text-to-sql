import { JSDOM } from 'jsdom';
import fs from 'fs';

// Read the HTML
const html = fs.readFileSync('app/ui/index.html', 'utf-8');

// Mock fetch for our endpoints
const mockTraces = [
  { task_id: 'tsk_test1', created_at: Date.now()/1000 - 1000 },
  { task_id: 'tsk_test2', created_at: Date.now()/1000 - 2000 }
];

const mockTraceDetails = [
  { step_id: 'plan', agent: 'planner', duration_ms: 1000, tokens_in: 100, tokens_out: 50, outcome: 'ok' },
  { step_id: 'investigate', agent: 'investigator', duration_ms: 2000, tokens_in: 200, tokens_out: 100, outcome: 'ok' }
];

global.fetch = async (url) => {
  if (url.includes('/traces') && !url.includes('tsk_')) {
    return { ok: true, json: async () => mockTraces };
  }
  if (url.includes('/traces/tsk_test1')) {
    return { ok: true, json: async () => mockTraceDetails };
  }
  return { ok: false, status: 404 };
};

// Create JSDOM with the HTML
const dom = new JSDOM(html, {
  url: 'http://localhost/ui/',
  pretendToBeVisual: true,
  runScripts: 'dangerously',
  resources: 'usable'
});

const { window } = dom;

// Polyfill needed methods
window.scrollTo = () => {};

// Wait for scripts to execute
await new Promise(resolve => setTimeout(resolve, 500));

console.log('=== JSDOM Test Started ===');
console.log('Initial view-traces display:', dom.window.getComputedStyle(dom.window.document.getElementById('view-traces')).display);

// Check if loadTraceList ran
const traceList = dom.window.document.getElementById('trace-list');
console.log('Trace list content (first 200 chars):', traceList.innerHTML.substring(0, 200));

// Check if trace cards rendered
const traceCards = dom.window.document.querySelectorAll('.trace-card');
console.log('Number of trace cards:', traceCards.length);

if (traceCards.length > 0) {
  console.log('First card task_id:', traceCards[0].querySelector('div').textContent);
  
  // Simulate click
  console.log('\n--- Simulating click on first trace card ---');
  const clickEvent = new dom.window.MouseEvent('click', { bubbles: true });
  traceCards[0].dispatchEvent(clickEvent);
  
  // Wait for async fetch and render
  await new Promise(resolve => setTimeout(resolve, 200));
  
  // Check trace details
  const traceDetails = dom.window.document.getElementById('trace-details');
  console.log('Trace details rendered:', traceDetails.innerHTML.substring(0, 300));
  
  // Check trace title
  const traceTitle = dom.window.document.getElementById('trace-title');
  console.log('Trace title:', traceTitle.textContent);
  
  // Check active state
  console.log('First card background:', dom.window.getComputedStyle(traceCards[0]).backgroundColor);
}

console.log('\n=== Test Complete ===');