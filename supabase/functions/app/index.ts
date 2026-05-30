const DASHBOARD_URL = "https://owandakin-coder.github.io/agent-analyst/";

Deno.serve(() => {
  return Response.redirect(DASHBOARD_URL, 302);
});
