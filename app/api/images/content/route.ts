import { env } from "cloudflare:workers";
const PREFIX="workflow-images/";
export async function GET(request:Request){const key=new URL(request.url).searchParams.get("key");if(!key?.startsWith(PREFIX)||!env.BUCKET)return new Response("Not found",{status:404});const object=await env.BUCKET.get(key);if(!object)return new Response("Not found",{status:404});const headers=new Headers();object.writeHttpMetadata(headers);headers.set("cache-control","private, max-age=3600");headers.set("x-content-type-options","nosniff");return new Response(object.body,{headers})}
