document.addEventListener('DOMContentLoaded', () => {
  const landingMenuButton = document.querySelector('.mobile-menu');
  const landingMenu = document.getElementById('landingMenu');
  const landingLinks = [...document.querySelectorAll('.landing-links a')];
  const closeLandingMenu = () => {
    landingMenu?.classList.remove('open');
    document.body.classList.remove('menu-open');
    landingMenuButton?.setAttribute('aria-expanded', 'false');
    const icon = landingMenuButton?.querySelector('i');
    if (icon) icon.className = 'bi bi-list';
  };
  landingMenuButton?.addEventListener('click', () => {
    const open = !landingMenu.classList.contains('open');
    landingMenu.classList.toggle('open', open);
    document.body.classList.toggle('menu-open', open);
    landingMenuButton.setAttribute('aria-expanded', String(open));
    landingMenuButton.querySelector('i').className = open ? 'bi bi-x-lg' : 'bi bi-list';
  });
  landingLinks.forEach(link => link.addEventListener('click', closeLandingMenu));
  document.addEventListener('keydown', event => { if (event.key === 'Escape') closeLandingMenu(); });
  if (landingLinks.length && 'IntersectionObserver' in window) {
    const linkById = Object.fromEntries(landingLinks.map(link => [link.hash.slice(1), link]));
    const observer = new IntersectionObserver(entries => {
      entries.filter(entry => entry.isIntersecting).forEach(entry => {
        landingLinks.forEach(link => link.classList.remove('active'));
        linkById[entry.target.id]?.classList.add('active');
      });
    }, {rootMargin:'-25% 0px -65% 0px'});
    document.querySelectorAll('main section[id], main#home').forEach(section => observer.observe(section));
  }
  const currentPath = window.location.pathname;
  if (currentPath.startsWith('/patient/')) {
    document.querySelectorAll('a, .quick-panel a, .stat-card').forEach(item => {
      if (/Doctor Mode|Clinical Data Prediction|Clinical Predictions|Patient Records|Prediction History|Analytics & Reports/.test(item.textContent)) item.remove();
    });
  }
  if (currentPath.startsWith('/doctor/')) {
    document.querySelectorAll('a').forEach(item => {
      if (item.textContent.trim() === 'Patient Mode') item.remove();
    });
  }
  const toast = (message, type = 'success') => {
    let box = document.getElementById('appToast');
    if (!box) {
      box = document.createElement('div');
      box.id = 'appToast';
      box.className = 'app-toast';
      box.innerHTML = '<i class="bi bi-check-circle-fill"></i><span></span>';
      document.body.appendChild(box);
    }
    box.classList.toggle('error', type === 'error');
    box.querySelector('span').textContent = message;
    box.classList.add('show');
    clearTimeout(window.oncoToastTimer);
    window.oncoToastTimer = setTimeout(() => box.classList.remove('show'), 3200);
  };

  document.querySelectorAll('.toggle-password').forEach(button => {
    button.addEventListener('click', () => {
      const input = document.getElementById(button.dataset.target);
      if (!input) return;
      input.type = input.type === 'password' ? 'text' : 'password';
      const icon = button.querySelector('i');
      if (icon) icon.className = input.type === 'password' ? 'bi bi-eye' : 'bi bi-eye-slash';
    });
  });

  document.querySelectorAll('.mode-option').forEach(option => {
    option.addEventListener('click', () => {
      document.querySelectorAll('.mode-option').forEach(item => item.classList.remove('selected'));
      option.classList.add('selected');
      const radio = option.querySelector('input[type="radio"]');
      if (radio) radio.checked = true;
    });
  });

  const loginForm = document.querySelector('form[action*="login"]');
  if (loginForm) {
    const queryMode = new URLSearchParams(location.search).get('mode');
    if (queryMode) document.querySelector(`.mode-option input[value="${queryMode}"]`)?.closest('.mode-option')?.click();
    if (new URLSearchParams(location.search).get('created') === 'true') toast('Account created — please log in');
    const loginError = document.body.dataset.loginError;
    if (loginError) {
      const feedback = document.createElement('div');
      feedback.className = 'login-feedback error';
      feedback.setAttribute('role', 'alert');
      feedback.innerHTML = '<i class="bi bi-exclamation-circle"></i><span></span>';
      feedback.querySelector('span').textContent = loginError;
      loginForm.before(feedback);
    }
  }

  const portalLinks = {
    'Patient Records':'patient-records', 'Prediction History':'prediction-history',
    'Reports':'reports', 'Resources':'resources', 'Profile Settings':'profile',
    'My Results':'analysis-history', 'Screening History':'analysis-history',
    'AI Explanation':'ai-explanations', 'Health Knowledge':'health-knowledge',
    'Profile':'profile', 'Settings':'settings', 'View Reports':'reports'
  };
  document.querySelectorAll('a[href="#"]').forEach(link => {
    const name = link.textContent.trim();
    const page = portalLinks[name];
    if (page) link.href = `/portal/${page}`;
    else link.addEventListener('click', event => { event.preventDefault(); toast(`${name || 'This option'} is ready.`); });
  });

  document.querySelectorAll('.topbar .icon-button:not(.bell)').forEach(button => {
    const sidebarId = button.getAttribute('aria-controls');
    const sidebar = (sidebarId && document.getElementById(sidebarId)) || document.querySelector('.dashboard-shell > .sidebar, .patient-side');
    const shell = button.closest('.dashboard-shell, .patient-app');
    if (!sidebar || !shell) return;
    button.setAttribute('aria-expanded', 'true');
    let overlay = shell.querySelector('.sidebar-menu-overlay');
    if (!overlay) {
      overlay = document.createElement('button');
      overlay.type = 'button';
      overlay.className = 'sidebar-menu-overlay';
      overlay.setAttribute('aria-label', 'Close navigation');
      shell.appendChild(overlay);
    }
    const setMenu = open => {
      const mobile = window.matchMedia('(max-width: 760px)').matches;
      if (mobile) {
        sidebar.classList.toggle('mobile-open', open);
        shell.classList.toggle('sidebar-mobile-open', open);
        document.body.classList.toggle('dashboard-menu-open', open);
      } else {
        shell.classList.toggle('sidebar-collapsed', !open);
        sidebar.classList.remove('mobile-open');
        shell.classList.remove('sidebar-mobile-open');
        document.body.classList.remove('dashboard-menu-open');
      }
      button.setAttribute('aria-expanded', String(open));
      button.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
      const icon = button.querySelector('i');
      if (icon) icon.className = 'bi bi-list';
    };
    button.addEventListener('click', () => {
      const mobile = window.matchMedia('(max-width: 760px)').matches;
      const currentlyOpen = mobile ? sidebar.classList.contains('mobile-open') : !shell.classList.contains('sidebar-collapsed');
      setMenu(!currentlyOpen);
    });
    overlay.addEventListener('click', () => setMenu(false));
    sidebar.querySelector('[data-sidebar-close]')?.addEventListener('click', () => setMenu(false));
    sidebar.querySelectorAll('a').forEach(link => link.addEventListener('click', () => {
      if (window.matchMedia('(max-width: 760px)').matches) setMenu(false);
    }));
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && sidebar.classList.contains('mobile-open')) setMenu(false);
    });
    window.addEventListener('resize', () => {
      const mobile = window.matchMedia('(max-width: 760px)').matches;
      if (mobile) setMenu(false);
      else setMenu(!shell.classList.contains('sidebar-collapsed'));
    });
    setMenu(!window.matchMedia('(max-width: 760px)').matches);
  });
  document.querySelectorAll('.bell').forEach(button => button.addEventListener('click', () => {
    button.querySelector('.badge')?.remove();
    toast('You have no unread notifications.');
  }));

  document.querySelectorAll('[data-demo-form]').forEach(form => form.addEventListener('submit', event => {
    event.preventDefault();
    toast('Your information was saved for this prototype session.');
    form.reset();
  }));

  const clinicalForm = document.getElementById('clinicalForm');
  if (clinicalForm) {
    const labels = ['T Stage','N Stage','M Stage','Grade','ER','PR','HER2','Ki-67','Age'];
    const names = ['tumor','node','metastasis','grade','er','pr','her2','ki67','age'];
    const updateSummary = () => {
      const values = Object.fromEntries(new FormData(clinicalForm));
      document.querySelectorAll('.summary-grid > div').forEach((chip, index) => {
        const value = values[names[index]] || '—';
        chip.querySelector('small').textContent = labels[index];
        chip.querySelector('b').textContent = index === 7 ? `${value}%` : index === 8 ? `${value} years` : value;
      });
    };
    clinicalForm.addEventListener('input', updateSummary);
    clinicalForm.addEventListener('reset', () => setTimeout(updateSummary));
    if (!clinicalForm.hasAttribute('data-manages-save')) {
      document.querySelector('.save-history')?.addEventListener('click', async () => {
        const data = Object.fromEntries(new FormData(clinicalForm));
        try {
          const response = await fetch('/api/clinical-records', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
          const result = await response.json();
          if (!response.ok) throw new Error(result.error || 'Unable to save record');
          toast(result.message);
        } catch (error) { toast(error.message, 'error'); }
      });
    }
  }

  const doctorCards = [...document.querySelectorAll('.doctor-card-ref')];
  const searchButton = document.querySelector('.search-ref > button');
  if (searchButton && doctorCards.length) {
    const searchInputs = document.querySelectorAll('.search-ref input');
    const doctorSearch = document.querySelector('.doctor-search');
    const searchDoctors = () => {
      const term = (doctorSearch?.value || '').toLowerCase().trim();
      const specialty = document.querySelector('.search-ref select')?.value || '';
      let visible = 0;
      doctorCards.forEach(card => {
        const matchesText = !term || card.textContent.toLowerCase().includes(term);
        const matchesSpecialty = !specialty || specialty === 'All Specialists' || card.textContent.includes(specialty);
        card.hidden = !(matchesText && matchesSpecialty);
        if (!card.hidden) visible++;
      });
      const counter = document.querySelector('.nearby-title h3 span');
      if (counter) counter.textContent = `${visible} Found`;
      toast(`${visible} specialist${visible === 1 ? '' : 's'} matched your search.`);
    };
    searchButton.addEventListener('click', searchDoctors);
    searchInputs.forEach(input => input.addEventListener('keydown', e => { if (e.key === 'Enter') searchDoctors(); }));
    document.querySelector('.search-ref select')?.addEventListener('change', searchDoctors);
    document.querySelectorAll('.doctor-card-ref aside button').forEach(button => button.addEventListener('click', () => {
      const card = button.closest('.doctor-card-ref');
      card.classList.toggle('profile-open');
      button.textContent = card.classList.contains('profile-open') ? 'Hide Profile' : 'View Profile';
      toast(`${card.classList.contains('profile-open') ? 'Showing' : 'Hiding'} ${card.querySelector('h4').childNodes[0].textContent.trim()}'s profile.`);
    }));
    document.querySelector('.view-more')?.addEventListener('click', event => {
      event.currentTarget.innerHTML = 'All available doctors are shown <i class="bi bi-check"></i>';
      toast('All available specialist profiles are displayed.');
    });
    document.querySelector('.map-toolbar')?.addEventListener('click', searchDoctors);
    document.querySelectorAll('.zoom button').forEach((button, index) => button.addEventListener('click', () => {
      const map = document.querySelector('.map-ref');
      const current = Number(map.dataset.zoom || 1);
      map.dataset.zoom = Math.max(.8, Math.min(1.25, current + (index ? -.1 : .1)));
      map.style.setProperty('--map-scale', map.dataset.zoom);
    }));
    const locateButton = document.querySelector('.bi-crosshair');
    const locateUser = () => {
      if (!navigator.geolocation) return toast('Geolocation is not supported by this browser.', 'error');
      navigator.geolocation.getCurrentPosition(position => {
        if (searchInputs[0]) searchInputs[0].value = `${position.coords.latitude.toFixed(4)}, ${position.coords.longitude.toFixed(4)}`;
        toast('Your current location was detected.');
      }, () => toast('Location permission was not granted.', 'error'));
    };
    locateButton?.addEventListener('click', locateUser);
    locateButton?.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); locateUser(); } });
  }

  document.querySelectorAll('.view-btn').forEach(button => button.addEventListener('click', () => toast('Patient record opened for review.')));
});
