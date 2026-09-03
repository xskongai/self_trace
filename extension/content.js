(() => {
  const SESSION_KEY = 'selftrace_session';
  const POSTS_PREFIX = 'selftrace_posts_';
  let collecting = false;
  let targetHandle = null;
  let posts = new Map();
  let observer = null;
  let scanTimer = null;
  let startedAt = null;

  function normalizeHandle(value) {
    return (value || '').replace(/^@/, '').trim().toLowerCase();
  }

  function currentProfileHandle() {
    const firstSegment = location.pathname.split('/').filter(Boolean)[0] || '';
    const blocked = new Set(['home','explore','notifications','messages','i','settings','search','compose','login','logout','signup','tos','privacy']);
    const handle = normalizeHandle(firstSegment);
    return handle && !blocked.has(handle) ? handle : null;
  }

  function postsKey(handle) { return `${POSTS_PREFIX}${normalizeHandle(handle)}`; }

  function statusUrlFromArticle(article) {
    const timeEl = article.querySelector('time');
    const anchor = timeEl?.closest('a[href*="/status/"]') || null;
    if (!anchor) return null;
    const rawHref = anchor.getAttribute('href');
    if (!rawHref) return null;
    const match = rawHref.match(/^\/([^/]+)\/status\/(\d+)/i);
    if (!match) return null;
    return {
      handle: normalizeHandle(match[1]),
      postId: match[2],
      url: new URL(`/${match[1]}/status/${match[2]}`, location.origin).href,
      createdAt: timeEl?.getAttribute('datetime') || null
    };
  }

  function extractImages(article) {
    const candidates = Array.from(article.querySelectorAll('img'))
      .map(img => ({ src: img.currentSrc || img.src || '', alt: img.alt || '' }))
      .filter(item => item.src)
      .filter(item => {
        try {
          const u = new URL(item.src);
          return u.hostname === 'pbs.twimg.com' && u.pathname.startsWith('/media/');
        } catch { return false; }
      });
    const result = [];
    const seen = new Set();
    for (const item of candidates) {
      let key = item.src;
      try {
        const u = new URL(item.src);
        u.searchParams.delete('name');
        key = u.toString();
      } catch {}
      if (seen.has(key)) continue;
      seen.add(key);
      result.push({ index: result.length, type: 'image', src: item.src, alt: item.alt });
    }
    return result;
  }

  function extractPost(article) {
    const status = statusUrlFromArticle(article);
    if (!status) return null;
    if (targetHandle && status.handle !== targetHandle) return null;
    const textNodes = Array.from(article.querySelectorAll('[data-testid="tweetText"]'));
    const text = textNodes.map(node => (node.innerText || '').trim()).filter(Boolean).join('\n');
    return {
      post_id: status.postId,
      author_handle: status.handle,
      created_at: status.createdAt,
      text,
      url: status.url,
      media: extractImages(article),
      collected_at: new Date().toISOString()
    };
  }

  async function persist() {
    if (!targetHandle) return;
    const ordered = Array.from(posts.values()).sort((a,b) => {
      const ta = a.created_at ? Date.parse(a.created_at) : 0;
      const tb = b.created_at ? Date.parse(b.created_at) : 0;
      return tb - ta;
    });
    const imageCount = ordered.reduce((sum,p) => sum + (p.media?.length || 0), 0);
    await chrome.storage.local.set({
      [postsKey(targetHandle)]: ordered,
      [SESSION_KEY]: {
        collecting,
        target_handle: targetHandle,
        post_count: ordered.length,
        image_count: imageCount,
        started_at: startedAt,
        updated_at: new Date().toISOString(),
        page_url: location.href
      }
    });
  }

  async function loadExisting(handle) {
    const key = postsKey(handle);
    const data = await chrome.storage.local.get(key);
    const existing = Array.isArray(data[key]) ? data[key] : [];
    posts = new Map(existing.map(post => [String(post.post_id), post]));
  }

  async function scanPage() {
    if (!collecting || !targetHandle) return;
    let changed = false;
    for (const article of Array.from(document.querySelectorAll('article'))) {
      const post = extractPost(article);
      if (!post) continue;
      const previous = posts.get(String(post.post_id));
      if (!previous) {
        posts.set(String(post.post_id), post);
        changed = true;
        continue;
      }
      const prevMedia = previous.media?.length || 0;
      const nextMedia = post.media?.length || 0;
      const prevText = previous.text || '';
      const nextText = post.text || '';
      if (nextMedia > prevMedia || nextText.length > prevText.length) {
        posts.set(String(post.post_id), { ...previous, ...post, collected_at: previous.collected_at || post.collected_at });
        changed = true;
      }
    }
    if (changed) await persist();
  }

  function scheduleScan() {
    clearTimeout(scanTimer);
    scanTimer = setTimeout(() => scanPage().catch(console.error), 350);
  }

  function startObserver() {
    if (observer) observer.disconnect();
    observer = new MutationObserver(scheduleScan);
    observer.observe(document.documentElement, { childList: true, subtree: true });
  }

  async function startCollection() {
    const handle = currentProfileHandle();
    if (!handle) throw new Error('Open your X profile page first, then start collection.');
    targetHandle = handle;
    await loadExisting(targetHandle);
    collecting = true;
    startedAt = startedAt || new Date().toISOString();
    startObserver();
    await scanPage();
    await persist();
    return getStatus();
  }

  async function stopCollection() {
    collecting = false;
    if (observer) { observer.disconnect(); observer = null; }
    clearTimeout(scanTimer);
    scanTimer = null;
    await persist();
    return getStatus();
  }

  function getStatus() {
    const values = Array.from(posts.values());
    return {
      collecting,
      target_handle: targetHandle,
      post_count: values.length,
      image_count: values.reduce((sum,p) => sum + (p.media?.length || 0), 0),
      started_at: startedAt,
      page_url: location.href
    };
  }

  async function clearCurrent() {
    const handle = targetHandle || currentProfileHandle();
    if (!handle) return;
    posts = new Map();
    startedAt = collecting ? new Date().toISOString() : null;
    await chrome.storage.local.remove(postsKey(handle));
    targetHandle = handle;
    await persist();
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    (async () => {
      try {
        switch (message?.type) {
          case 'SELFTRACE_START': sendResponse({ok:true,status:await startCollection()}); break;
          case 'SELFTRACE_STOP': sendResponse({ok:true,status:await stopCollection()}); break;
          case 'SELFTRACE_STATUS': sendResponse({ok:true,status:getStatus()}); break;
          case 'SELFTRACE_SCAN_NOW': await scanPage(); sendResponse({ok:true,status:getStatus()}); break;
          case 'SELFTRACE_CLEAR': await clearCurrent(); sendResponse({ok:true,status:getStatus()}); break;
          default: sendResponse({ok:false,error:'Unknown SelfTrace message.'});
        }
      } catch (error) {
        sendResponse({ok:false,error:error?.message || String(error)});
      }
    })();
    return true;
  });
})();
