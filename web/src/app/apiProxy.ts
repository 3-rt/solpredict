const API_BASE_URL =
  process.env.SOLPREDICT_API_URL || process.env.API_URL || process.env.NEXT_PUBLIC_API_URL;

type ProxyOptions = {
  path: string;
  request: Request;
  method: "GET" | "POST";
};

function jsonResponse(body: object, status: number) {
  return Response.json(body, {
    status,
    headers: {
      "cache-control": "no-store",
    },
  });
}

export async function proxyApiRequest({ path, request, method }: ProxyOptions) {
  if (!API_BASE_URL) {
    return jsonResponse({ error: "SolPredict API URL is not configured." }, 503);
  }

  const incomingUrl = new URL(request.url);
  const upstreamUrl = new URL(path, API_BASE_URL);
  upstreamUrl.search = incomingUrl.search;

  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);

  try {
    const upstreamResponse = await fetch(upstreamUrl, {
      method,
      headers,
      body: method === "POST" ? await request.text() : undefined,
      cache: "no-store",
    });

    const responseHeaders = new Headers(upstreamResponse.headers);
    responseHeaders.set("cache-control", "no-store");
    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      headers: responseHeaders,
    });
  } catch {
    return jsonResponse({ error: "SolPredict API is unreachable." }, 502);
  }
}
