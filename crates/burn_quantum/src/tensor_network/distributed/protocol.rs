use super::*;
use crate::transport::{self, DeadlineStream};
use serde::{Deserialize, Serialize};
use std::net::TcpStream;
use std::time::Instant;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) enum Command {
    Inspect,
    Read { epoch: u64, indices: Vec<usize> },
    Prepare(Update),
    Commit { epoch: u64 },
    Abort { epoch: u64 },
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub version: u32,
    pub identity: StateIdentity,
    pub capability: String,
    pub command: Command,
}
#[derive(Serialize, Deserialize)]
pub(super) enum Reply {
    Meta(Metadata),
    Sites(Vec<(usize, SiteData)>),
    Ack,
    Rejected(String),
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Response {
    pub version: u32,
    pub identity: StateIdentity,
    pub reply: Reply,
}
pub(super) fn call(
    endpoint: &WorkerEndpoint,
    identity: &StateIdentity,
    command: Command,
    limits: StateNetworkLimits,
) -> Result<(Reply, usize)> {
    limits.validate()?;
    identity.validate()?;
    let end = Instant::now()
        .checked_add(limits.io_timeout)
        .ok_or(Error::Overflow)?;
    let stream = TcpStream::connect_timeout(&endpoint.address, limits.io_timeout).map_err(err)?;
    let mut stream = DeadlineStream::new(stream, end)?;
    let request = Request {
        version: 1,
        identity: identity.clone(),
        capability: endpoint.capability.clone(),
        command,
    };
    transport::write(&mut stream, &request, limits.max_frame_bytes)?;
    let response: Response = transport::read(&mut stream, limits.max_frame_bytes)?;
    if response.version != 1 || response.identity != *identity {
        return Err(err("response identity/version differs"));
    }
    // Both objects have already passed bounded framing before this diagnostic encoding.
    let bytes = serde_json::to_vec(&request)
        .map_err(err)?
        .len()
        .checked_add(serde_json::to_vec(&response).map_err(err)?.len())
        .and_then(|x| x.checked_add(16))
        .ok_or(Error::Overflow)?;
    match response.reply {
        Reply::Rejected(e) => Err(err(e)),
        reply => Ok((reply, bytes)),
    }
}
